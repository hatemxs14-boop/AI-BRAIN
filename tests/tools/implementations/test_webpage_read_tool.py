"""
Tests for the real read_webpage executor (core.tools.implementations.
webpage_read_tool), including the SSRF-defense rules in
_assert_safe_public_url that keep it from ever fetching a private/
internal network address.

No real network access is used: every test either injects a fake
HTTP layer (via the executor's own `http_get=` parameter) or patches
`socket.getaddrinfo` on the module to control what address a hostname
"resolves" to, exactly as tests/tools/implementations/
test_web_search_tool.py fakes its HTTP layer for the same reason.
"""
from __future__ import annotations

import os
import shutil
import socket
import tempfile
from unittest.mock import patch

import pytest
import requests

from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.implementations.webpage_read_tool import (
    READ_WEBPAGE_TOOL,
    READ_WEBPAGE_TOOL_ID,
    _assert_safe_public_url,
    create_webpage_read_executor,
)
from core.tools.registry.tool_registry import ToolRegistry


PERMISSIONS_FILE = "core/security/schemas/permissions.json"

PUBLIC_IP = "93.184.216.34"


class _FakeResponse:

    def __init__(
        self,
        *,
        status_code=200,
        content=b"",
        headers=None,
    ):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


def _patched_resolution(ip: str):
    """
    Patch socket.getaddrinfo (as imported by webpage_read_tool) to
    resolve any hostname to `ip`, so tests never depend on real DNS.
    """

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (ip, port),
            )
        ]

    return patch(
        "core.tools.implementations.webpage_read_tool.socket.getaddrinfo",
        side_effect=fake_getaddrinfo,
    )


# ---------------------------------------------------------------------
# ToolDefinition contract
# ---------------------------------------------------------------------

def test_read_webpage_tool_definition_registers_cleanly():
    registry = ToolRegistry()
    registry.register(READ_WEBPAGE_TOOL)

    assert registry.contains(READ_WEBPAGE_TOOL_ID)

    tool = registry.get(READ_WEBPAGE_TOOL_ID)
    assert tool.resource == "webpage"
    assert tool.action == "read"
    assert tool.scope == "public_web"
    assert tool.risk_level == "LOW"


# ---------------------------------------------------------------------
# Executor construction
# ---------------------------------------------------------------------

def test_create_executor_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout"):
        create_webpage_read_executor(timeout=0)


def test_create_executor_rejects_non_positive_max_response_bytes():
    with pytest.raises(ValueError, match="max_response_bytes"):
        create_webpage_read_executor(max_response_bytes=0)


def test_create_executor_rejects_non_positive_max_content_chars():
    with pytest.raises(ValueError, match="max_content_chars"):
        create_webpage_read_executor(max_content_chars=-1)


# ---------------------------------------------------------------------
# _assert_safe_public_url: the core SSRF-defense property.
# ---------------------------------------------------------------------

def test_assert_safe_public_url_rejects_disallowed_scheme():
    with pytest.raises(PermissionError, match="scheme"):
        _assert_safe_public_url("ftp://example.com/file.txt")


def test_assert_safe_public_url_rejects_missing_hostname():
    with pytest.raises(ValueError, match="hostname"):
        _assert_safe_public_url("https:///no-host-here")


def test_assert_safe_public_url_rejects_unresolvable_host():
    def fake_getaddrinfo(host, port, *args, **kwargs):
        raise socket.gaierror("Name or service not known")

    with patch(
        "core.tools.implementations.webpage_read_tool.socket.getaddrinfo",
        side_effect=fake_getaddrinfo,
    ):
        with pytest.raises(RuntimeError, match="Could not resolve"):
            _assert_safe_public_url("https://no-such-host.invalid/")


def test_assert_safe_public_url_rejects_loopback_address():
    with _patched_resolution("127.0.0.1"):
        with pytest.raises(PermissionError, match="private/"):
            _assert_safe_public_url("http://localhost/")


def test_assert_safe_public_url_rejects_link_local_metadata_address():
    with _patched_resolution("169.254.169.254"):
        with pytest.raises(PermissionError, match="private/"):
            _assert_safe_public_url("http://metadata.example/latest")


def test_assert_safe_public_url_rejects_private_network_address():
    with _patched_resolution("10.0.0.5"):
        with pytest.raises(PermissionError, match="private/"):
            _assert_safe_public_url("http://internal.example/")


def test_assert_safe_public_url_allows_public_address():
    with _patched_resolution(PUBLIC_IP):
        _assert_safe_public_url("https://example.com/article")


# ---------------------------------------------------------------------
# Executor behavior (fake HTTP layer + patched resolution)
# ---------------------------------------------------------------------

def test_executor_extracts_title_and_text_and_strips_script_style():
    html = (
        b"<html><head><title>  My Article  </title>"
        b"<style>body { color: red; }</style></head>"
        b"<body><h1>Heading</h1><p>First paragraph.</p>"
        b"<script>alert('nope');</script>"
        b"<p>Second paragraph.</p></body></html>"
    )

    def fake_get(url, **kwargs):
        return _FakeResponse(content=html)

    executor = create_webpage_read_executor(http_get=fake_get)

    with _patched_resolution(PUBLIC_IP):
        result = executor(url="https://example.com/article")

    assert result["url"] == "https://example.com/article"
    assert result["title"] == "My Article"
    assert "Heading" in result["content"]
    assert "First paragraph." in result["content"]
    assert "Second paragraph." in result["content"]
    assert "alert(" not in result["content"]
    assert "color: red" not in result["content"]
    assert result["truncated"] is False


def test_executor_rejects_declared_content_length_over_limit():
    def fake_get(url, **kwargs):
        return _FakeResponse(
            content=b"<html></html>",
            headers={"Content-Length": "999999999"},
        )

    executor = create_webpage_read_executor(
        http_get=fake_get,
        max_response_bytes=1000,
    )

    with _patched_resolution(PUBLIC_IP):
        with pytest.raises(ValueError, match="exceeding"):
            executor(url="https://example.com/huge")


def test_executor_truncates_oversized_body_without_content_length():
    big_html = b"<html><body>" + (b"x" * 5000) + b"</body></html>"

    def fake_get(url, **kwargs):
        return _FakeResponse(content=big_html)

    executor = create_webpage_read_executor(
        http_get=fake_get,
        max_response_bytes=100,
    )

    with _patched_resolution(PUBLIC_IP):
        result = executor(url="https://example.com/big")

    assert result["truncated"] is True


def test_executor_truncates_extracted_text_to_max_content_chars():
    html = b"<html><body>" + (b"word " * 1000) + b"</body></html>"

    def fake_get(url, **kwargs):
        return _FakeResponse(content=html)

    executor = create_webpage_read_executor(
        http_get=fake_get,
        max_content_chars=50,
    )

    with _patched_resolution(PUBLIC_IP):
        result = executor(url="https://example.com/long")

    assert len(result["content"]) <= 50
    assert result["truncated"] is True


def test_executor_raises_on_non_200_status():
    def fake_get(url, **kwargs):
        return _FakeResponse(status_code=404, content=b"not found")

    executor = create_webpage_read_executor(http_get=fake_get)

    with _patched_resolution(PUBLIC_IP):
        with pytest.raises(RuntimeError, match="status=404"):
            executor(url="https://example.com/missing")


def test_executor_raises_on_request_exception():
    def fake_get(url, **kwargs):
        raise requests.ConnectionError("network is unreachable")

    executor = create_webpage_read_executor(http_get=fake_get)

    with _patched_resolution(PUBLIC_IP):
        with pytest.raises(RuntimeError, match="fetch failed"):
            executor(url="https://example.com/anything")


def test_executor_rejects_empty_url():
    executor = create_webpage_read_executor(
        http_get=lambda *a, **k: _FakeResponse(content=b"<html></html>")
    )

    with pytest.raises(ValueError, match="non-empty string"):
        executor(url="   ")


def test_executor_rejects_url_resolving_to_private_address():
    executor = create_webpage_read_executor(
        http_get=lambda *a, **k: _FakeResponse(content=b"SHOULD NOT BE FETCHED")
    )

    with _patched_resolution("192.168.1.1"):
        with pytest.raises(PermissionError, match="private/"):
            executor(url="http://internal.example/secret")


# ---------------------------------------------------------------------
# Full ToolGateway integration (real security stack)
# ---------------------------------------------------------------------

def test_full_gateway_execution_succeeds_without_approval():
    audit_dir = tempfile.mkdtemp()
    try:
        registry = ToolRegistry()
        registry.register(READ_WEBPAGE_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        def fake_get(url, **kwargs):
            return _FakeResponse(
                content=b"<html><head><title>T</title></head>"
                b"<body><p>Evidence goes here.</p></body></html>"
            )

        gateway.register_executor(
            tool_id=READ_WEBPAGE_TOOL_ID,
            executor=create_webpage_read_executor(http_get=fake_get),
        )

        with _patched_resolution(PUBLIC_IP):
            result = gateway.execute(
                subject="research_agent",
                tool_id=READ_WEBPAGE_TOOL_ID,
                tool_kwargs={"url": "https://example.com/evidence"},
            )

        assert result.status == "SUCCESS"
        assert result.security_decision.decision.value == "ALLOW"

        (artifact,) = result.artifacts
        assert artifact["title"] == "T"
        assert "Evidence goes here." in artifact["content"]
    finally:
        shutil.rmtree(audit_dir)


def test_full_gateway_execution_surfaces_ssrf_rejection_as_tool_error():
    audit_dir = tempfile.mkdtemp()
    try:
        registry = ToolRegistry()
        registry.register(READ_WEBPAGE_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        gateway.register_executor(
            tool_id=READ_WEBPAGE_TOOL_ID,
            executor=create_webpage_read_executor(
                http_get=lambda *a, **k: _FakeResponse(
                    content=b"SHOULD NOT BE FETCHED"
                )
            ),
        )

        with _patched_resolution("127.0.0.1"):
            result = gateway.execute(
                subject="research_agent",
                tool_id=READ_WEBPAGE_TOOL_ID,
                tool_kwargs={"url": "http://localhost/admin"},
            )

        assert result.status == "ERROR"
        assert "private/" in result.artifacts[0]
    finally:
        shutil.rmtree(audit_dir)

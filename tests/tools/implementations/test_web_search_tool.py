"""
Tests for the real web_search executor (core.tools.implementations.
web_search_tool), backed by Serper.dev.

This sandbox has no outbound internet access, and even where a test
environment does, a unit test should not depend on a live third-party
API or a real API key -- the same reasoning this project already
applies to the Claude/OpenAI provider tests. Every test here injects a
fake HTTP layer (either via the executor's own `http_post=` parameter,
or by patching `requests.post` on the module for the full-stack test)
instead of making a real network call.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
import requests

from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.implementations.web_search_tool import (
    WEB_SEARCH_TOOL,
    WEB_SEARCH_TOOL_ID,
    create_serper_web_search_executor,
)
from core.tools.registry.tool_registry import ToolRegistry


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


class _FakeResponse:

    def __init__(self, *, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload configured")
        return self._payload


def _build_gateway(audit_dir: str) -> tuple[ToolGateway, ToolRegistry]:
    registry = ToolRegistry()
    registry.register(WEB_SEARCH_TOOL)

    security = SecurityDecisionPoint(
        PERMISSIONS_FILE,
        audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)

    return gateway, registry


# ---------------------------------------------------------------------
# ToolDefinition contract
# ---------------------------------------------------------------------

def test_web_search_tool_definition_registers_cleanly():
    registry = ToolRegistry()
    registry.register(WEB_SEARCH_TOOL)

    assert registry.contains(WEB_SEARCH_TOOL_ID)

    tool = registry.get(WEB_SEARCH_TOOL_ID)
    assert tool.resource == "web_search"
    assert tool.action == "search"
    assert tool.scope == "public_web"
    assert tool.risk_level == "LOW"


# ---------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------

def test_create_executor_raises_without_api_key():
    original = os.environ.pop("SERPER_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="Serper.dev API key"):
            create_serper_web_search_executor(api_key=None)
    finally:
        if original is not None:
            os.environ["SERPER_API_KEY"] = original


def test_create_executor_uses_env_var_fallback():
    original = os.environ.get("SERPER_API_KEY")
    os.environ["SERPER_API_KEY"] = "from-env-var"
    try:
        captured_kwargs = {}

        def fake_post(url, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeResponse(payload={"organic": []})

        executor = create_serper_web_search_executor(
            api_key=None,
            http_post=fake_post,
        )
        executor(query="test")

        assert captured_kwargs["headers"]["X-API-KEY"] == "from-env-var"
    finally:
        if original is None:
            os.environ.pop("SERPER_API_KEY", None)
        else:
            os.environ["SERPER_API_KEY"] = original


# ---------------------------------------------------------------------
# Executor behavior (fake HTTP layer)
# ---------------------------------------------------------------------

def test_executor_parses_successful_response_and_truncates_results():
    organic = [
        {
            "title": f"Result {i}",
            "link": f"https://example.com/{i}",
            "snippet": f"Snippet {i}",
        }
        for i in range(10)
    ]

    def fake_post(url, **kwargs):
        assert url == "https://google.serper.dev/search"
        assert kwargs["json"] == {"q": "AI agents"}
        return _FakeResponse(payload={"organic": organic})

    executor = create_serper_web_search_executor(
        api_key="test-key",
        max_results=3,
        http_post=fake_post,
    )

    result = executor(query="AI agents")

    assert result["query"] == "AI agents"
    assert len(result["results"]) == 3
    assert result["results"][0] == {
        "title": "Result 0",
        "link": "https://example.com/0",
        "snippet": "Snippet 0",
    }


def test_executor_defaults_missing_fields_to_empty_string():
    def fake_post(url, **kwargs):
        return _FakeResponse(
            payload={"organic": [{"title": "Only a title"}]}
        )

    executor = create_serper_web_search_executor(
        api_key="test-key",
        http_post=fake_post,
    )

    result = executor(query="anything")

    assert result["results"] == [
        {"title": "Only a title", "link": "", "snippet": ""}
    ]


def test_executor_handles_missing_or_non_list_organic_field():
    def fake_post(url, **kwargs):
        return _FakeResponse(payload={})

    executor = create_serper_web_search_executor(
        api_key="test-key",
        http_post=fake_post,
    )

    result = executor(query="anything")

    assert result == {"query": "anything", "results": []}


def test_executor_raises_on_non_200_status():
    def fake_post(url, **kwargs):
        return _FakeResponse(status_code=401, text="Unauthorized")

    executor = create_serper_web_search_executor(
        api_key="bad-key",
        http_post=fake_post,
    )

    with pytest.raises(RuntimeError, match="status=401"):
        executor(query="anything")


def test_executor_raises_on_invalid_json():
    def fake_post(url, **kwargs):
        return _FakeResponse(payload=None)

    executor = create_serper_web_search_executor(
        api_key="test-key",
        http_post=fake_post,
    )

    with pytest.raises(RuntimeError, match="not valid JSON"):
        executor(query="anything")


def test_executor_raises_on_request_exception():
    def fake_post(url, **kwargs):
        raise requests.ConnectionError("network is unreachable")

    executor = create_serper_web_search_executor(
        api_key="test-key",
        http_post=fake_post,
    )

    with pytest.raises(RuntimeError, match="request failed"):
        executor(query="anything")


def test_executor_rejects_empty_query():
    executor = create_serper_web_search_executor(
        api_key="test-key",
        http_post=lambda *a, **k: _FakeResponse(payload={"organic": []}),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        executor(query="   ")


# ---------------------------------------------------------------------
# Full ToolGateway integration (real security stack, fake HTTP layer)
# ---------------------------------------------------------------------

def test_full_gateway_execution_succeeds_without_approval():
    audit_dir = tempfile.mkdtemp()
    try:
        gateway, _registry = _build_gateway(audit_dir)

        organic = [
            {
                "title": "AI Agents Explained",
                "link": "https://example.com/ai-agents",
                "snippet": "An overview of AI agents.",
            }
        ]

        def fake_post(url, **kwargs):
            return _FakeResponse(payload={"organic": organic})

        gateway.register_executor(
            tool_id=WEB_SEARCH_TOOL_ID,
            executor=create_serper_web_search_executor(
                api_key="test-key",
                http_post=fake_post,
            ),
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id=WEB_SEARCH_TOOL_ID,
            tool_kwargs={"query": "AI agents"},
        )

        assert result.status == "SUCCESS"
        assert result.security_decision.decision.value == "ALLOW"

        (artifact,) = result.artifacts
        assert artifact["query"] == "AI agents"
        assert artifact["results"][0]["title"] == "AI Agents Explained"
    finally:
        shutil.rmtree(audit_dir)


def test_full_gateway_execution_surfaces_search_errors_as_tool_error():
    audit_dir = tempfile.mkdtemp()
    try:
        gateway, _registry = _build_gateway(audit_dir)

        def failing_post(url, **kwargs):
            raise requests.Timeout("timed out")

        gateway.register_executor(
            tool_id=WEB_SEARCH_TOOL_ID,
            executor=create_serper_web_search_executor(
                api_key="test-key",
                http_post=failing_post,
            ),
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id=WEB_SEARCH_TOOL_ID,
            tool_kwargs={"query": "AI agents"},
        )

        assert result.status == "ERROR"
        assert "request failed" in result.artifacts[0]
    finally:
        shutil.rmtree(audit_dir)


def test_patching_requests_post_directly_also_works():
    """
    Confirms the default (no http_post override) code path -- the one
    actually used in production via build_research_agent -- calls
    `requests.post` as imported in this module, so patching it there
    is a valid way for a higher-level test (e.g. the research_agent
    wiring tests) to fake the network without reaching into this
    module's internals.
    """
    with patch(
        "core.tools.implementations.web_search_tool.requests.post"
    ) as mock_post:
        mock_post.return_value = _FakeResponse(payload={"organic": []})

        executor = create_serper_web_search_executor(api_key="test-key")
        result = executor(query="anything")

        assert result == {"query": "anything", "results": []}
        mock_post.assert_called_once()

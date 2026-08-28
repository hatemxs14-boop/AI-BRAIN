from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.parse import urlsplit

import requests

from core.tools.registry.tool_registry import ToolDefinition


# ---------------------------------------------------------------------
# Real read_webpage tool for research_agent.
#
# web_search (core.tools.implementations.web_search_tool) only ever
# returns title/link/snippet -- it deliberately never fetches or reads
# the linked pages themselves (see that tool's own `purpose` string).
# For a research agent, a search result's snippet is rarely enough to
# actually verify a claim or extract real evidence; the natural,
# missing second half of "search the web" is "read what a search
# result actually says". This module is that second half: a real,
# read-only HTTP GET of a single public URL, with its HTML reduced to
# plain readable text (title + body text, scripts/styles stripped).
#
# RESEARCH_AGENT.md's "Tools" section allows "approved read-only
# search tools" and nothing about command execution or writes -- this
# tool performs exactly one read-only network GET per call and never
# writes anywhere, matching that boundary the same way web_search and
# read_document already do.
#
# Because this tool accepts an arbitrary caller-supplied URL rather
# than a fixed, trusted endpoint (unlike web_search's hardcoded
# Serper.dev endpoint), it needs its own defense against SSRF: a
# request pointed at "http://169.254.169.254/" or "http://localhost/"
# or an internal 10.x/192.168.x address must never be allowed to
# succeed, or this tool becomes a way to probe or read from internal
# infrastructure under the guise of "reading a webpage". See
# _assert_safe_public_url below for exactly what is blocked and its
# one honestly-documented limitation (DNS-rebinding TOCTOU).
#
# There was no existing permission entry for this in permissions.json
# -- a new research_agent:webpage:read:public_web / risk_level=LOW
# entry was added alongside this module (LOW to match web_search's own
# risk_level for the same reason: both are read-only public-internet
# fetches with no side effects, and RiskEngine classifies action="read"
# with an unclassified resource as LOW -- see risk_engine.py's "Safe
# read-only operations" branch).
# ---------------------------------------------------------------------

READ_WEBPAGE_TOOL_ID = "read_webpage"

ALLOWED_SCHEMES = ("http", "https")

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_MAX_CONTENT_CHARS = 20_000


READ_WEBPAGE_TOOL = ToolDefinition(
    id=READ_WEBPAGE_TOOL_ID,
    name="Read Webpage",
    purpose=(
        "Fetch one public webpage by URL over HTTP(S) and return its "
        "readable text content (title plus body text, with scripts, "
        "styles, and markup stripped). Read-only: never writes "
        "anything, and never follows a URL that resolves to a "
        "private/internal network address."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "truncated": {"type": "boolean"},
        },
        "required": ["url", "title", "content", "truncated"],
        "additionalProperties": False,
    },
    permissions=(
        "research_agent:webpage:read:public_web",
    ),
    resource="webpage",
    action="read",
    scope="public_web",
    risk_level="LOW",
    error_handling={
        "retryable": True,
        "max_retries": 2,
        "on_failure": (
            "Surface the fetch error to the agent as a failed tool "
            "result. A URL rejected as unsafe (bad scheme, private/"
            "internal address) or malformed will not become fetchable "
            "by retrying the identical request; a transient network "
            "failure (timeout, connection error) may succeed on retry."
        ),
    },
)


class _TextExtractor(HTMLParser):
    """
    Minimal HTML-to-text reducer: stdlib-only (no new dependency), by
    design not a full sanitizer or renderer. Collects the page
    <title> separately, drops <script>/<style>/<noscript> contents
    entirely, and keeps every other tag's text data as whitespace-
    joined chunks. Good enough to turn a webpage into something an
    LLM can read as evidence; not a general-purpose HTML parser.
    """

    _SKIPPED_TAGS = frozenset({"script", "style", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return

        if self._in_title:
            self.title += data
            return

        stripped = data.strip()

        if stripped:
            self._chunks.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def _assert_safe_public_url(url: str) -> None:
    """
    Raise if `url` is not safe to fetch as a "public webpage".

    Rejects (in order): a scheme other than http/https; a URL with no
    hostname; a hostname that fails to resolve; and a hostname whose
    resolved address is loopback, link-local (this also covers the
    common 169.254.169.254 cloud-metadata address), private, reserved,
    unspecified, or multicast.

    KNOWN LIMITATION, documented rather than solved: this resolves the
    hostname once, here, to decide safety, but the actual HTTP request
    below performs its own independent DNS resolution. A malicious
    server under attacker control could in principle answer this
    lookup with a public IP and a later lookup (during the real
    request) with a private one ("DNS rebinding"). Closing that gap
    completely requires pinning the exact resolved address for the
    real request too (e.g. a custom requests transport), which is a
    meaningfully larger change for a research tool whose worst case
    today is an internal GET request, not code execution or data
    exfiltration. Revisit if this tool is ever given write access or
    exposed to less-trusted callers.
    """

    parsed = urlsplit(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise PermissionError(
            f"URL '{url}' uses scheme '{parsed.scheme or ''}', which "
            f"is not allowed; only {ALLOWED_SCHEMES} may be fetched."
        )

    hostname = parsed.hostname

    if not hostname:
        raise ValueError(f"URL '{url}' does not have a hostname.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        address_infos = socket.getaddrinfo(hostname, port)
    except socket.gaierror as exc:
        raise RuntimeError(
            f"Could not resolve host '{hostname}' for URL '{url}': "
            f"{exc}"
        ) from exc

    for _family, _type, _proto, _canonname, sockaddr in address_infos:
        ip = ipaddress.ip_address(sockaddr[0])

        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_reserved
            or ip.is_unspecified
            or ip.is_multicast
        ):
            raise PermissionError(
                f"URL '{url}' resolves to '{sockaddr[0]}', a private/"
                "internal address; only public webpages may be "
                "fetched."
            )


def create_webpage_read_executor(
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    http_get: Callable[..., Any] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """
    Build a real executor for READ_WEBPAGE_TOOL.

    The returned callable has the exact signature ToolGateway.execute()
    calls with (`executor(**tool_kwargs)`, i.e. `executor(url=...)` for
    this tool's input_schema).

    `http_get` is an injection point for tests: it defaults to
    `requests.get`, but a test can supply a fake to exercise this
    executor's parsing/error-handling logic without making a real
    network call -- this sandbox has no outbound internet access, and
    even where it does, a unit test should not depend on a live
    third-party webpage (the same reasoning already applied to
    web_search_tool's tests).

    `max_response_bytes` caps how much of the raw HTTP body is kept:
    if the server reports a larger `Content-Length`, the fetch is
    rejected before reading the body; if it doesn't (or lies), the
    body is truncated to this many bytes after the fact -- a
    best-effort cap, not a true streaming limit (see this function's
    docstring companion, _assert_safe_public_url, for a similar
    documented trade-off). `max_content_chars` separately caps the
    *extracted text* returned to the agent, independent of the raw
    body size, matching RESEARCH_AGENT.md's own "Context Budget"
    guidance to avoid flooding the agent with large inline content.
    """

    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive number.")

    if (
        not isinstance(max_response_bytes, int)
        or max_response_bytes <= 0
    ):
        raise ValueError("max_response_bytes must be a positive integer.")

    if not isinstance(max_content_chars, int) or max_content_chars <= 0:
        raise ValueError("max_content_chars must be a positive integer.")

    get = http_get if http_get is not None else requests.get

    def _execute(url: str) -> dict[str, Any]:

        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string.")

        _assert_safe_public_url(url)

        try:
            response = get(url, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Webpage fetch failed for '{url}': {exc}"
            ) from exc

        status_code = getattr(response, "status_code", None)

        if status_code != 200:
            raise RuntimeError(
                "Webpage fetch returned a non-200 response "
                f"(status={status_code!r}) for '{url}'."
            )

        headers = getattr(response, "headers", None) or {}
        declared_length_raw = headers.get(
            "Content-Length"
        ) or headers.get("content-length")

        if declared_length_raw is not None:
            try:
                declared_length = int(declared_length_raw)
            except (TypeError, ValueError):
                declared_length = None

            if (
                declared_length is not None
                and declared_length > max_response_bytes
            ):
                raise ValueError(
                    f"Webpage '{url}' declares a body of "
                    f"{declared_length} bytes, exceeding the "
                    f"{max_response_bytes}-byte fetch limit."
                )

        body = getattr(response, "content", None)

        if body is None:
            body = str(getattr(response, "text", "")).encode(
                "utf-8",
                errors="replace",
            )

        truncated = False

        if len(body) > max_response_bytes:
            body = body[:max_response_bytes]
            truncated = True

        html_text = body.decode("utf-8", errors="replace")

        parser = _TextExtractor()
        parser.feed(html_text)

        text = parser.get_text()

        if len(text) > max_content_chars:
            text = text[:max_content_chars]
            truncated = True

        return {
            "url": url,
            "title": parser.title.strip(),
            "content": text,
            "truncated": truncated,
        }

    return _execute

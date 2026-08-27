from __future__ import annotations

import os
from typing import Any, Callable

import requests

from core.tools.registry.tool_registry import ToolDefinition


# ---------------------------------------------------------------------
# Real web_search tool for research_agent.
#
# This is the first genuinely real (non-mocked) tool implementation in
# AI-BRAIN. Every prior appearance of a "web_search" tool in this
# codebase (tests/agents/test_real_agent_llm_loop.py included) used a
# fake in-memory executor -- the point of the exercise there was to
# prove the security/agent plumbing worked end-to-end, not to actually
# search anything. This module backs the same tool contract with a
# real HTTP call to Serper.dev (Google search results), so
# research_agent can do real research.
#
# research_agent's permission for this exact resource/action/scope
# already exists in permissions.json (subject=research_agent,
# resource=web_search, action=search, scope=public_web, risk_level=
# LOW) -- it was added purely as documentation/test fixture data
# across Passes 1-5 with no real executor ever backing it. This module
# is what finally makes that permission mean something.
# ---------------------------------------------------------------------

SERPER_ENDPOINT = "https://google.serper.dev/search"

WEB_SEARCH_TOOL_ID = "web_search"

DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT_SECONDS = 10.0


WEB_SEARCH_TOOL = ToolDefinition(
    id=WEB_SEARCH_TOOL_ID,
    name="Web Search",
    purpose=(
        "Search the public web (via Serper.dev / Google search "
        "results) for read-only research. Returns a ranked list of "
        "titles, links, and snippets; it does not fetch or read the "
        "linked pages themselves."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
            },
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "link": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                    "required": ["title", "link", "snippet"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["query", "results"],
        "additionalProperties": False,
    },
    permissions=(
        "research_agent:web_search:search:public_web",
    ),
    resource="web_search",
    action="search",
    scope="public_web",
    risk_level="LOW",
    error_handling={
        "retryable": True,
        "max_retries": 2,
        "on_failure": (
            "Surface the search API error to the agent as a failed "
            "tool result. Never fabricate search results when the "
            "real search fails -- an empty or errored result must "
            "stay visibly empty/errored."
        ),
    },
)


def create_serper_web_search_executor(
    *,
    api_key: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    http_post: Callable[..., Any] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """
    Build a real executor for WEB_SEARCH_TOOL, backed by Serper.dev.

    The returned callable has the exact signature ToolGateway.execute()
    calls with (`executor(**tool_kwargs)`, i.e. `executor(query=...)`
    for this tool's input_schema).

    `api_key` falls back to the `SERPER_API_KEY` environment variable
    when omitted -- the same pattern this project already uses for
    `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`. Raises immediately (at
    executor-construction time, not on first use) when no key is
    available, so a misconfigured deployment fails loudly at startup
    rather than on the agent's first tool call.

    `http_post` is an injection point for tests: it defaults to
    `requests.post`, but a test can supply a fake to exercise this
    executor's parsing/error-handling logic without making a real
    network call (this sandbox has no outbound internet access, and
    even where it does, tests should not depend on a live third-party
    API or a real API key -- the same reasoning already applied to the
    OpenAI/Claude provider tests in this project).
    """

    resolved_key = (
        api_key
        if api_key is not None
        else os.environ.get("SERPER_API_KEY")
    )

    if not isinstance(resolved_key, str) or not resolved_key.strip():
        raise ValueError(
            "A Serper.dev API key is required to build the web_search "
            "executor. Pass api_key= explicitly or set the "
            "SERPER_API_KEY environment variable. Get a key at "
            "https://serper.dev."
        )

    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError("max_results must be a positive integer.")

    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive number.")

    post = http_post if http_post is not None else requests.post

    def _execute(query: str) -> dict[str, Any]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        try:
            response = post(
                SERPER_ENDPOINT,
                headers={
                    "X-API-KEY": resolved_key,
                    "Content-Type": "application/json",
                },
                json={"q": query},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Serper.dev web search request failed: {exc}"
            ) from exc

        status_code = getattr(response, "status_code", None)

        if status_code != 200:
            body_preview = str(
                getattr(response, "text", "")
            )[:500]

            raise RuntimeError(
                "Serper.dev web search returned a non-200 response "
                f"(status={status_code!r}): {body_preview!r}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Serper.dev web search returned a response that is "
                "not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError(
                "Serper.dev web search returned an unexpected "
                f"response shape: {type(payload).__name__}."
            )

        organic = payload.get("organic", [])

        if not isinstance(organic, list):
            organic = []

        results: list[dict[str, str]] = []

        for entry in organic[:max_results]:

            if not isinstance(entry, dict):
                continue

            results.append(
                {
                    "title": str(entry.get("title", "")),
                    "link": str(entry.get("link", "")),
                    "snippet": str(entry.get("snippet", "")),
                }
            )

        return {
            "query": query,
            "results": results,
        }

    return _execute

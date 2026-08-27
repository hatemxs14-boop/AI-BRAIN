from __future__ import annotations

from pathlib import Path
from typing import Any

from core.agents.agent_core import (
    AgentCore,
    AgentIdentity,
)

from core.agents.agent_loop import (
    AgentExecutionLoop,
    AgentLoopResult,
)

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.agents.llm_decision_engine import (
    LLMDecisionEngine,
)

from core.agents.tool_interface import (
    AgentToolInterface,
)

from core.llm.llm_client import (
    LLMClient,
)

from core.security.engine.security_decision import (
    SecurityDecisionPoint,
)

from core.tools.engine.tool_gateway import (
    ToolGateway,
)

from core.tools.implementations.document_read_tool import (
    READ_DOCUMENT_TOOL,
    READ_DOCUMENT_TOOL_ID,
    create_document_read_executor,
)

from core.tools.implementations.web_search_tool import (
    WEB_SEARCH_TOOL,
    WEB_SEARCH_TOOL_ID,
    create_serper_web_search_executor,
)

from core.tools.registry.tool_registry import (
    ToolRegistry,
)

from core.tools.runtime.tool_runtime import (
    ToolRuntime,
)


# ---------------------------------------------------------------------
# research_agent -- wired to its real, currently-approved toolset.
#
# core/agents/RESEARCH_AGENT.md has described this agent since early in
# the project; until now it existed only as that spec document plus
# scattered test fixtures (permissions.json entries, mocked executors
# in tests/agents/test_real_agent_llm_loop.py) exercising the security
# plumbing, never as a runnable agent with real tools behind it. This
# module is the actual wiring: it assembles the full stack (Registry ->
# Gateway -> Security Layer -> real Executors -> Runtime -> Agent) the
# same way every test in this project's history has, except the
# executors here really do something instead of returning a canned
# string.
#
# Real tools wired here (both LOW risk_level -- auto-execute, no
# approval required):
#
#   web_search      resource=web_search   action=search  scope=public_web
#                    -> core.tools.implementations.web_search_tool
#                       (real Serper.dev / Google search call)
#
#   read_document   resource=document     action=read    scope=workspace
#                    -> core.tools.implementations.document_read_tool
#                       (real, sandboxed plain-text/Markdown file read)
#
# Deliberately NOT wired: permissions.json also grants research_agent a
# HIGH-risk permission for resource=shell/action=execute/scope=
# workspace (present since Pass 1's test fixtures, unrelated to this
# build phase). RESEARCH_AGENT.md's own "Tools" section lists only
# "approved read-only search tools" and "approved document/file
# reading tools" as allowed, explicitly forbids "destructive
# filesystem tools", and never mentions command execution as an
# allowed capability at all -- shell execution is not "destructive"
# by definition, but it is also nowhere on the agent's allowed list,
# and there is no legitimate research-agent use case this build phase
# identified for it. Rather than either (a) silently building a real
# shell executor the spec never asked for, or (b) editing
# permissions.json to remove a grant that may exist for a future,
# different subject/tool, this module simply never registers a
# ToolDefinition for resource=shell here: no ToolDefinition means
# nothing is exposed through discover_tools_for_subject(), so the
# latent permission grants nothing today. This should be resolved
# explicitly (wire a real, scoped shell tool, or drop the permission
# entry) before research_agent is given any task that could plausibly
# call for command execution.
# ---------------------------------------------------------------------

RESEARCH_AGENT_SUBJECT = "research_agent"

DEFAULT_PERMISSIONS_PATH = "core/security/schemas/permissions.json"
DEFAULT_DOCUMENTS_ROOT = "workspace/research_documents"


def build_research_agent(
    *,
    documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT,
    serper_api_key: str | None = None,
    permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH,
    audit_log_path: str | None = None,
) -> AgentCore:
    """
    Assemble a fully wired research_agent AgentCore: real tools,
    real security stack, ready to receive a task and run through an
    AgentExecutionLoop.

    `serper_api_key` falls back to the `SERPER_API_KEY` environment
    variable (see create_serper_web_search_executor) -- raises
    immediately if neither is available, so a misconfigured deployment
    fails at build time rather than on the agent's first search.

    `documents_root` must already exist as a directory (see
    create_document_read_executor) -- defaults to the
    workspace/research_documents/ sandbox shipped in this repo.
    """

    registry = ToolRegistry()

    registry.register(WEB_SEARCH_TOOL)
    registry.register(READ_DOCUMENT_TOOL)

    security_kwargs: dict[str, Any] = {}

    if audit_log_path is not None:
        security_kwargs["audit_log_path"] = audit_log_path

    security = SecurityDecisionPoint(
        str(permissions_path),
        **security_kwargs,
    )

    gateway = ToolGateway(
        security=security,
        registry=registry,
    )

    gateway.register_executor(
        tool_id=WEB_SEARCH_TOOL_ID,
        executor=create_serper_web_search_executor(
            api_key=serper_api_key,
        ),
    )

    gateway.register_executor(
        tool_id=READ_DOCUMENT_TOOL_ID,
        executor=create_document_read_executor(
            documents_root,
        ),
    )

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    interface = AgentToolInterface(
        runtime=runtime,
    )

    identity = AgentIdentity(
        subject=RESEARCH_AGENT_SUBJECT,
        name="Research Agent",
        purpose=(
            "Conduct structured, read-only research and return "
            "evidence-backed findings. See "
            "core/agents/RESEARCH_AGENT.md for the full role "
            "specification."
        ),
    )

    return AgentCore(
        identity=identity,
        tools=interface,
    )


def run_research_agent(
    task: str,
    *,
    llm_client: LLMClient | None = None,
    decision_engine: AgentDecisionEngine | None = None,
    documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT,
    serper_api_key: str | None = None,
    permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH,
    audit_log_path: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_steps: int = 10,
) -> AgentLoopResult:
    """
    Convenience entry point: build the research_agent, start `task`,
    run it through an AgentExecutionLoop to a terminal result, and
    return that result.

    Provide exactly one of `llm_client` (wraps it in a
    LLMDecisionEngine using `model`/`temperature`/`max_tokens`) or a
    pre-built `decision_engine` (e.g. a deterministic engine for
    testing, or an LLMDecisionEngine already configured some other
    way).
    """

    if decision_engine is None:

        if llm_client is None:
            raise ValueError(
                "Either llm_client or decision_engine must be "
                "provided."
            )

        decision_engine = LLMDecisionEngine(
            llm_client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    agent = build_research_agent(
        documents_root=documents_root,
        serper_api_key=serper_api_key,
        permissions_path=permissions_path,
        audit_log_path=audit_log_path,
    )

    agent.start_task(task)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=decision_engine,
        max_steps=max_steps,
    )

    return loop.run()

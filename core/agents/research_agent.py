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

from core.tools.implementations.webpage_read_tool import (
    READ_WEBPAGE_TOOL,
    READ_WEBPAGE_TOOL_ID,
    create_webpage_read_executor,
)

from core.tools.implementations.write_research_findings_tool import (
    WRITE_RESEARCH_FINDINGS_TOOL,
    WRITE_RESEARCH_FINDINGS_TOOL_ID,
    create_write_research_findings_executor,
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
# Real tools wired here:
#
#   web_search      resource=web_search        action=search scope=public_web
#                    -> core.tools.implementations.web_search_tool
#                       (real Serper.dev / Google search call)
#                       LOW risk -- auto-executes, no approval required.
#
#   read_document   resource=document          action=read   scope=workspace
#                    -> core.tools.implementations.document_read_tool
#                       (real, sandboxed plain-text/Markdown file read)
#                       LOW risk -- auto-executes, no approval required.
#
#   read_webpage    resource=webpage           action=read   scope=public_web
#                    -> core.tools.implementations.webpage_read_tool
#                       (real HTTP GET of one public URL, HTML reduced
#                       to plain text -- closes the gap web_search
#                       leaves open, since web_search only ever returns
#                       title/link/snippet and never the page content
#                       itself; see that module's own docstring for its
#                       SSRF-defense design and one documented
#                       limitation)
#                       LOW risk -- auto-executes, no approval required.
#
#   write_research_findings  resource=research_findings action=write
#                             scope=workspace
#                    -> core.tools.implementations.write_research_findings_tool
#                       (real, sandboxed, write-once persistence of a
#                       research finding -- the "write research
#                       findings when explicitly authorized" capability
#                       RESEARCH_AGENT.md's Memory Access section
#                       allows)
#                       HIGH risk, "policy" approval -- every call
#                       returns APPROVAL_REQUIRED unless the caller
#                       supplies an explicit, attributed approval; see
#                       that module's own docstring for why this is
#                       the one tool here that is NOT auto-executing.
#
# Resolved contradiction: permissions.json used to also grant
# research_agent a HIGH-risk permission for resource=shell/
# action=execute/scope=workspace (present since Pass 1's test
# fixtures, unrelated to this build phase). RESEARCH_AGENT.md's own
# "Tools" section lists only "approved read-only search tools" and
# "approved document/file reading tools" as allowed, explicitly
# forbids "destructive filesystem tools", and never mentions command
# execution as an allowed capability at all -- there was no legitimate
# research-agent use case for shell execution, so rather than build a
# real shell executor the spec never asked for, the shell permission
# entry has been removed from permissions.json entirely. research_agent
# now holds no shell-related permission of any kind: even a direct,
# hand-built ToolInvocation for resource=shell against this subject is
# denied by AuthorizationEngine at the policy layer, on top of this
# module never registering a ToolDefinition for it (which already kept
# it out of discover_tools_for_subject()). No further action is needed
# here; if a future build phase adds real command-execution capability
# for research_agent, it should be introduced as a new, explicitly
# scoped/whitelisted tool with its own permission entry and its own
# executor, not by restoring this one.
# ---------------------------------------------------------------------

RESEARCH_AGENT_SUBJECT = "research_agent"

DEFAULT_PERMISSIONS_PATH = "core/security/schemas/permissions.json"
DEFAULT_DOCUMENTS_ROOT = "workspace/research_documents"
DEFAULT_FINDINGS_ROOT = "workspace/research_findings"


def build_research_agent(
    *,
    documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT,
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
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

    `findings_root` must likewise already exist as a directory (see
    create_write_research_findings_executor) -- defaults to the
    workspace/research_findings/ sandbox shipped in this repo. Every
    write_research_findings call still requires explicit approval
    regardless of this path (see this module's own docstring); this
    only controls WHERE an approved write is allowed to land.
    """

    registry = ToolRegistry()

    registry.register(WEB_SEARCH_TOOL)
    registry.register(READ_DOCUMENT_TOOL)
    registry.register(READ_WEBPAGE_TOOL)
    registry.register(WRITE_RESEARCH_FINDINGS_TOOL)

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

    gateway.register_executor(
        tool_id=READ_WEBPAGE_TOOL_ID,
        executor=create_webpage_read_executor(),
    )

    gateway.register_executor(
        tool_id=WRITE_RESEARCH_FINDINGS_TOOL_ID,
        executor=create_write_research_findings_executor(
            findings_root,
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
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
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
        findings_root=findings_root,
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

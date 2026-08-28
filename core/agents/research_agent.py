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

from core.memory.memory_store import (
    MemoryStore,
)

from core.policies.policy_engine import (
    PolicyEngine,
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

from core.tools.implementations.read_project_memory_tool import (
    READ_PROJECT_MEMORY_TOOL,
    READ_PROJECT_MEMORY_TOOL_ID,
    create_read_project_memory_executor,
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
#   read_project_memory  resource=project_memory action=read
#                        scope=workspace
#                    -> core.tools.implementations.read_project_memory_tool
#                       (real, keyword-search read access to the
#                       Memory Layer -- Build Phase 14 -- closing the
#                       "read approved project memory" capability
#                       RESEARCH_AGENT.md's Memory Access section has
#                       declared since this spec was first written,
#                       with nothing implementing it until now)
#                       LOW risk -- auto-executes, no approval
#                       required. Its results are untrusted context,
#                       never fed automatically into anything this
#                       module builds; see that tool's own docstring
#                       and core/memory/MEMORY_SPEC.md.
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
#
# Agent Constraints check (Build Phase 9): RESEARCH_AGENT_DECLARED_TOOL_IDS
# below is this module's own explicit statement of the tool ids
# RESEARCH_AGENT.md's "Tools > Allowed" section actually names as
# allowed. build_research_agent() checks the real tool ids it just
# registered against this declared set via
# PolicyEngine.evaluate_agent_scope() (core/policies/policy_engine.py)
# and raises immediately if they ever diverge -- see that method's own
# docstring for exactly what this does and does not cover. This never
# fires under normal operation (the registrations below are the only
# ones this function ever performs); it exists to catch a future code
# change that silently registers a tool this agent's own spec doesn't
# declare, per POLICY_SPEC.md's Agent Constraints ("never silently
# expand their scope").
#
# Agent Constraints check (Build Phase 10): build_research_agent() also
# calls PolicyEngine.evaluate_agent_permission_alignment() right after
# constructing its SecurityDecisionPoint, comparing each registered
# tool's (resource, action, scope) against permissions.json's real
# grants for research_agent -- a second, config-side slice of the same
# "operate only within declared responsibilities" bullet the tool-id
# check above covers from the code side. See that method's own
# docstring (core/policies/policy_engine.py) for exactly what this
# catches and why it is safe under the "never so strict it can't
# execute" constraint (build-time only, never a runtime gate).
# ---------------------------------------------------------------------

RESEARCH_AGENT_SUBJECT = "research_agent"

DEFAULT_PERMISSIONS_PATH = "core/security/schemas/permissions.json"
DEFAULT_DOCUMENTS_ROOT = "workspace/research_documents"
DEFAULT_FINDINGS_ROOT = "workspace/research_findings"
DEFAULT_MEMORY_STORE_PATH = "workspace/project_memory/memory.jsonl"

RESEARCH_AGENT_DECLARED_TOOL_IDS = frozenset(
    {
        WEB_SEARCH_TOOL_ID,
        READ_DOCUMENT_TOOL_ID,
        READ_WEBPAGE_TOOL_ID,
        WRITE_RESEARCH_FINDINGS_TOOL_ID,
        READ_PROJECT_MEMORY_TOOL_ID,
    }
)


def build_research_agent(
    *,
    documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT,
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
    memory_store_path: str | Path = DEFAULT_MEMORY_STORE_PATH,
    serper_api_key: str | None = None,
    permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH,
    audit_log_path: str | None = None,
    policy_engine: PolicyEngine | None = None,
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

    `memory_store_path` (Build Phase 14) is the JSON-Lines file backing
    read_project_memory's MemoryStore -- defaults to
    workspace/project_memory/memory.jsonl. Unlike documents_root/
    findings_root, this path does not need to already exist:
    MemoryStore creates its parent directory itself (mirrors
    AuditLogger's own precedent), so a fresh deployment with no memory
    recorded yet still builds successfully and simply returns no
    search results until something writes to the store.

    `policy_engine` defaults to a fresh PolicyEngine() -- injected
    mainly for tests that want to substitute or inspect it (see this
    module's own docstring for the Agent Constraints check it performs
    here).
    """

    if policy_engine is None:
        policy_engine = PolicyEngine()

    registry = ToolRegistry()

    registry.register(WEB_SEARCH_TOOL)
    registry.register(READ_DOCUMENT_TOOL)
    registry.register(READ_WEBPAGE_TOOL)
    registry.register(WRITE_RESEARCH_FINDINGS_TOOL)
    registry.register(READ_PROJECT_MEMORY_TOOL)

    scope_evaluation = policy_engine.evaluate_agent_scope(
        subject=RESEARCH_AGENT_SUBJECT,
        declared_tool_ids=RESEARCH_AGENT_DECLARED_TOOL_IDS,
        actual_tool_ids={tool.id for tool in registry.list_tools()},
    )

    if not scope_evaluation.within_scope:
        raise ValueError(
            "research_agent's build_research_agent() registered "
            f"tool(s) {sorted(scope_evaluation.unauthorized_tool_ids)} "
            "that are not declared in "
            "RESEARCH_AGENT_DECLARED_TOOL_IDS -- this means the code "
            "has silently expanded past what "
            "core/agents/RESEARCH_AGENT.md's own 'Tools' section "
            "declares as allowed. See POLICY_SPEC.md's Agent "
            "Constraints ('never silently expand their scope')."
        )

    security_kwargs: dict[str, Any] = {}

    if audit_log_path is not None:
        security_kwargs["audit_log_path"] = audit_log_path

    security = SecurityDecisionPoint(
        str(permissions_path),
        **security_kwargs,
    )

    permission_alignment = policy_engine.evaluate_agent_permission_alignment(
        subject=RESEARCH_AGENT_SUBJECT,
        tool_grants_needed={
            (tool.resource, tool.action, tool.scope)
            for tool in registry.list_tools()
        },
        security_grants_present={
            (permission.get("resource"), permission.get("action"), permission.get("scope"))
            for permission in security.authorization_engine.policy.get(
                "permissions", []
            )
            if isinstance(permission, dict)
            and permission.get("subject") == RESEARCH_AGENT_SUBJECT
        },
    )

    if not permission_alignment.aligned:
        raise ValueError(
            "research_agent's build_research_agent() has drifted from "
            f"{permissions_path}: missing grant(s) "
            f"{sorted(permission_alignment.missing_grants)} (a "
            "registered tool needs these but permissions.json never "
            "grants them -- every real call would be DENIED), extra "
            f"grant(s) {sorted(permission_alignment.extra_grants)} (a "
            "standing permission no registered tool needs at all). See "
            "POLICY_SPEC.md's Agent Constraints ('operate only within "
            "declared responsibilities')."
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

    gateway.register_executor(
        tool_id=READ_PROJECT_MEMORY_TOOL_ID,
        executor=create_read_project_memory_executor(
            MemoryStore(str(memory_store_path)),
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
    memory_store_path: str | Path = DEFAULT_MEMORY_STORE_PATH,
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
        memory_store_path=memory_store_path,
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

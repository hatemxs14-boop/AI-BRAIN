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

from core.policies.policy_engine import (
    PolicyEngine,
)

from core.security.engine.security_decision import (
    SecurityDecisionPoint,
)

from core.tools.engine.tool_gateway import (
    ToolGateway,
)

from core.tools.implementations.read_research_findings_tool import (
    READ_RESEARCH_FINDINGS_TOOL,
    READ_RESEARCH_FINDINGS_TOOL_ID,
    create_read_research_findings_executor,
)

from core.tools.implementations.write_report_tool import (
    WRITE_REPORT_TOOL,
    WRITE_REPORT_TOOL_ID,
    create_write_report_executor,
)

from core.tools.registry.tool_registry import (
    ToolRegistry,
)

from core.tools.runtime.tool_runtime import (
    ToolRuntime,
)


# ---------------------------------------------------------------------
# writer_agent -- Build Phase 8's second real agent.
#
# core/agents/WRITER_AGENT.md describes this agent's role: a writing/
# synthesis specialist, distinct from research_agent's evidence-
# gathering role, that reads research_agent's already-persisted
# findings (workspace/research_findings/, written via research_agent's
# own write_research_findings tool) and drafts a written report from
# them -- the natural second half of a research -> write pipeline
# AGENT_REGISTRY.md's own "Collaboration" section describes
# ("Multiple agents may be selected when the task contains independent
# domains ... specialized expertise is required").
#
# Real tools wired here:
#
#   read_research_findings  resource=research_findings action=read
#                            scope=workspace
#                    -> core.tools.implementations.
#                       read_research_findings_tool (real, sandboxed
#                       plain-text/Markdown read of a finding
#                       research_agent already wrote)
#                       LOW risk -- auto-executes, no approval
#                       required. A new, writer_agent-only permission
#                       (never research_agent's own document:read:
#                       workspace grant -- a distinct subject never
#                       inherits another subject's permission
#                       implicitly; see that tool module's own
#                       docstring).
#
#   write_report             resource=report action=write
#                             scope=workspace
#                    -> core.tools.implementations.write_report_tool
#                       (real, sandboxed, write-once persistence of a
#                       report -- the "publish a written report when
#                       explicitly authorized" capability
#                       WRITER_AGENT.md's Memory Access section
#                       allows)
#                       HIGH risk, "policy" approval -- every call
#                       returns APPROVAL_REQUIRED unless the caller
#                       supplies an explicit, attributed approval,
#                       structurally identical to research_agent's own
#                       write_research_findings gate (Build Phase 3).
#
# writer_agent holds no read access to workspace/research_documents/
# (research_agent's own source-material sandbox) and no write access
# to workspace/research_findings/ (research_agent's own output) --
# each agent's tools and permissions are its own, per AGENT_REGISTRY.md's
# Boundaries ("must not access memory outside its declared scope").
#
# Agent Constraints check (Build Phase 9): WRITER_AGENT_DECLARED_TOOL_IDS
# below is this module's own explicit statement of the tool ids
# WRITER_AGENT.md's "Tools > Allowed" section actually names as
# allowed. build_writer_agent() checks the real tool ids it just
# registered against this declared set via
# PolicyEngine.evaluate_agent_scope() (core/policies/policy_engine.py)
# and raises immediately if they ever diverge -- see that method's own
# docstring, and core/agents/research_agent.py's identical check, for
# exactly what this does and does not cover.
#
# Agent Constraints check (Build Phase 10): build_writer_agent() also
# calls PolicyEngine.evaluate_agent_permission_alignment() right after
# constructing its SecurityDecisionPoint -- the same config-side
# alignment check research_agent.py's own module docstring describes,
# applied here to writer_agent's own two tools/grants.
# ---------------------------------------------------------------------

WRITER_AGENT_SUBJECT = "writer_agent"

DEFAULT_PERMISSIONS_PATH = "core/security/schemas/permissions.json"
DEFAULT_FINDINGS_ROOT = "workspace/research_findings"
DEFAULT_REPORTS_ROOT = "workspace/reports"

WRITER_AGENT_DECLARED_TOOL_IDS = frozenset(
    {
        READ_RESEARCH_FINDINGS_TOOL_ID,
        WRITE_REPORT_TOOL_ID,
    }
)


def build_writer_agent(
    *,
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH,
    audit_log_path: str | None = None,
    policy_engine: PolicyEngine | None = None,
) -> AgentCore:
    """
    Assemble a fully wired writer_agent AgentCore: real tools, real
    security stack, ready to receive a task and run through an
    AgentExecutionLoop.

    `findings_root` must already exist as a directory (see
    create_read_research_findings_executor) -- defaults to the same
    workspace/research_findings/ sandbox research_agent's own
    write_research_findings tool writes into, so writer_agent reads
    exactly what research_agent has actually persisted.

    `reports_root` must likewise already exist as a directory (see
    create_write_report_executor) -- defaults to the
    workspace/reports/ sandbox shipped in this repo. Every write_report
    call still requires explicit approval regardless of this path (see
    this module's own docstring); this only controls WHERE an approved
    write is allowed to land.

    `policy_engine` defaults to a fresh PolicyEngine() -- injected
    mainly for tests that want to substitute or inspect it (see this
    module's own docstring for the Agent Constraints check it performs
    here).
    """

    if policy_engine is None:
        policy_engine = PolicyEngine()

    registry = ToolRegistry()

    registry.register(READ_RESEARCH_FINDINGS_TOOL)
    registry.register(WRITE_REPORT_TOOL)

    scope_evaluation = policy_engine.evaluate_agent_scope(
        subject=WRITER_AGENT_SUBJECT,
        declared_tool_ids=WRITER_AGENT_DECLARED_TOOL_IDS,
        actual_tool_ids={tool.id for tool in registry.list_tools()},
    )

    if not scope_evaluation.within_scope:
        raise ValueError(
            "writer_agent's build_writer_agent() registered tool(s) "
            f"{sorted(scope_evaluation.unauthorized_tool_ids)} that "
            "are not declared in WRITER_AGENT_DECLARED_TOOL_IDS -- "
            "this means the code has silently expanded past what "
            "core/agents/WRITER_AGENT.md's own 'Tools' section "
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
        subject=WRITER_AGENT_SUBJECT,
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
            and permission.get("subject") == WRITER_AGENT_SUBJECT
        },
    )

    if not permission_alignment.aligned:
        raise ValueError(
            "writer_agent's build_writer_agent() has drifted from "
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
        tool_id=READ_RESEARCH_FINDINGS_TOOL_ID,
        executor=create_read_research_findings_executor(
            findings_root,
        ),
    )

    gateway.register_executor(
        tool_id=WRITE_REPORT_TOOL_ID,
        executor=create_write_report_executor(
            reports_root,
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
        subject=WRITER_AGENT_SUBJECT,
        name="Writer Agent",
        purpose=(
            "Synthesize already-persisted research findings into a "
            "written report and publish it when explicitly approved. "
            "See core/agents/WRITER_AGENT.md for the full role "
            "specification."
        ),
    )

    return AgentCore(
        identity=identity,
        tools=interface,
    )


def run_writer_agent(
    task: str,
    *,
    llm_client: LLMClient | None = None,
    decision_engine: AgentDecisionEngine | None = None,
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH,
    audit_log_path: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_steps: int = 10,
) -> AgentLoopResult:
    """
    Convenience entry point: build writer_agent, start `task`, run it
    through an AgentExecutionLoop to a terminal result, and return
    that result.

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

    agent = build_writer_agent(
        findings_root=findings_root,
        reports_root=reports_root,
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

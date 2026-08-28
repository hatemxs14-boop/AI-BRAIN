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

from core.tools.implementations.read_report_tool import (
    READ_REPORT_TOOL,
    READ_REPORT_TOOL_ID,
    create_read_report_executor,
)

from core.tools.implementations.read_research_findings_tool import (
    READ_RESEARCH_FINDINGS_TOOL,
    READ_RESEARCH_FINDINGS_TOOL_ID,
    create_read_research_findings_executor,
)

from core.tools.registry.tool_registry import (
    ToolRegistry,
)

from core.tools.runtime.tool_runtime import (
    ToolRuntime,
)


# ---------------------------------------------------------------------
# reviewer_agent -- Build Phase 11's third real agent.
#
# core/agents/REVIEWER_AGENT.md describes this agent's role: a
# read-only, independent-verification specialist -- distinct from both
# research_agent's evidence-gathering role and writer_agent's
# synthesis/publishing role -- that reads a report writer_agent has
# already published (workspace/reports/, written via writer_agent's
# own write_report tool) together with the research findings that
# report claims to be based on (workspace/research_findings/, written
# via research_agent's own write_research_findings tool), and reports
# which claims are actually supported. This is the third leg of a
# research -> write -> review pipeline, matching AGENT_REGISTRY.md's
# own "Collaboration" section ("independent verification provides
# meaningful value" is one of its four explicit reasons multiple
# agents may be selected).
#
# This does not replace or extend Kernel._verify()
# (core/kernel/kernel.py): that check remains exactly what its own
# docstring says it is -- a narrow, generic consistency check that a
# COMPLETED result's last tool call actually succeeded, applying to
# every agent equally. reviewer_agent is a genuine, deliberate
# capability an orchestrator or human can invoke to get real,
# independent, content-level verification of one specific report --
# the "dedicated verification subsystem (independently re-checking
# claims, not just consistency-checking the agent's own last result)"
# KernelVerification's own docstring names as future work remains
# exactly that: this agent is a new tool for a human/orchestrator to
# use, not a new Kernel-level gate every run passes through.
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
#                       required. This is the SAME ToolDefinition
#                       writer_agent already uses (same resource, same
#                       root, same trust boundary) -- its own
#                       `permissions` tuple now names both writer_agent
#                       and reviewer_agent explicitly (never
#                       implicitly; see that module's own docstring),
#                       and permissions.json separately grants each
#                       subject its own real authorization entry.
#
#   read_report              resource=report action=read
#                             scope=workspace
#                    -> core.tools.implementations.read_report_tool
#                       (real, sandboxed plain-text/Markdown read of a
#                       report writer_agent already published --
#                       previously there was no way to read a
#                       published report back at all)
#                       LOW risk -- auto-executes, no approval
#                       required. A new, reviewer_agent-only
#                       permission -- never writer_agent's own
#                       write_report grant (a distinct subject never
#                       inherits another subject's permission
#                       implicitly).
#
# reviewer_agent holds no write access to either workspace/
# research_findings/ or workspace/reports/, and no read access to
# workspace/research_documents/ (research_agent's own source-material
# sandbox) -- each agent's tools and permissions are its own, per
# AGENT_REGISTRY.md's Boundaries ("must not access memory outside its
# declared scope"). Unlike research_agent and writer_agent,
# reviewer_agent has no HIGH-risk/approval-gated tool at all: every
# tool it holds is a LOW-risk read, matching its own spec's
# description as a purely read-only, independent-verification
# specialist that never persists anything.
#
# Agent Constraints check (Build Phase 9): REVIEWER_AGENT_DECLARED_TOOL_IDS
# below is this module's own explicit statement of the tool ids
# REVIEWER_AGENT.md's "Tools > Allowed" section actually names as
# allowed. build_reviewer_agent() checks the real tool ids it just
# registered against this declared set via
# PolicyEngine.evaluate_agent_scope() (core/policies/policy_engine.py)
# and raises immediately if they ever diverge -- see that method's own
# docstring, and core/agents/research_agent.py's/writer_agent.py's
# identical checks, for exactly what this does and does not cover.
#
# Agent Constraints check (Build Phase 10): build_reviewer_agent() also
# calls PolicyEngine.evaluate_agent_permission_alignment() right after
# constructing its SecurityDecisionPoint -- the same config-side
# alignment check research_agent.py's own module docstring describes,
# applied here to reviewer_agent's own two tools/grants.
# ---------------------------------------------------------------------

REVIEWER_AGENT_SUBJECT = "reviewer_agent"

DEFAULT_PERMISSIONS_PATH = "core/security/schemas/permissions.json"
DEFAULT_FINDINGS_ROOT = "workspace/research_findings"
DEFAULT_REPORTS_ROOT = "workspace/reports"

REVIEWER_AGENT_DECLARED_TOOL_IDS = frozenset(
    {
        READ_RESEARCH_FINDINGS_TOOL_ID,
        READ_REPORT_TOOL_ID,
    }
)


def build_reviewer_agent(
    *,
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH,
    audit_log_path: str | None = None,
    policy_engine: PolicyEngine | None = None,
) -> AgentCore:
    """
    Assemble a fully wired reviewer_agent AgentCore: real tools, real
    security stack, ready to receive a task and run through an
    AgentExecutionLoop.

    `findings_root` must already exist as a directory (see
    create_read_research_findings_executor) -- defaults to the same
    workspace/research_findings/ sandbox research_agent's own
    write_research_findings tool writes into and writer_agent already
    reads from, so reviewer_agent reads exactly the same findings
    writer_agent had available.

    `reports_root` must likewise already exist as a directory (see
    create_read_report_executor) -- defaults to the same
    workspace/reports/ sandbox writer_agent's own write_report tool
    writes into, so reviewer_agent reads exactly what writer_agent has
    actually published.

    `policy_engine` defaults to a fresh PolicyEngine() -- injected
    mainly for tests that want to substitute or inspect it (see this
    module's own docstring for the Agent Constraints checks it
    performs here).
    """

    if policy_engine is None:
        policy_engine = PolicyEngine()

    registry = ToolRegistry()

    registry.register(READ_RESEARCH_FINDINGS_TOOL)
    registry.register(READ_REPORT_TOOL)

    scope_evaluation = policy_engine.evaluate_agent_scope(
        subject=REVIEWER_AGENT_SUBJECT,
        declared_tool_ids=REVIEWER_AGENT_DECLARED_TOOL_IDS,
        actual_tool_ids={tool.id for tool in registry.list_tools()},
    )

    if not scope_evaluation.within_scope:
        raise ValueError(
            "reviewer_agent's build_reviewer_agent() registered "
            f"tool(s) {sorted(scope_evaluation.unauthorized_tool_ids)} "
            "that are not declared in "
            "REVIEWER_AGENT_DECLARED_TOOL_IDS -- this means the code "
            "has silently expanded past what "
            "core/agents/REVIEWER_AGENT.md's own 'Tools' section "
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
        subject=REVIEWER_AGENT_SUBJECT,
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
            and permission.get("subject") == REVIEWER_AGENT_SUBJECT
        },
    )

    if not permission_alignment.aligned:
        raise ValueError(
            "reviewer_agent's build_reviewer_agent() has drifted from "
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
        tool_id=READ_REPORT_TOOL_ID,
        executor=create_read_report_executor(
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
        subject=REVIEWER_AGENT_SUBJECT,
        name="Reviewer Agent",
        purpose=(
            "Independently verify an already-published report "
            "against the research findings it claims to be based on. "
            "See core/agents/REVIEWER_AGENT.md for the full role "
            "specification."
        ),
    )

    return AgentCore(
        identity=identity,
        tools=interface,
    )


def run_reviewer_agent(
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
    Convenience entry point: build reviewer_agent, start `task`, run
    it through an AgentExecutionLoop to a terminal result, and return
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

    agent = build_reviewer_agent(
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

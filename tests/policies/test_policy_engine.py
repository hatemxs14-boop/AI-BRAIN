"""
Tests for core.policies.policy_engine (Policy Layer v1).

See core/policies/policy_engine.py's own module docstring for exactly
what this v1 implements vs. deliberately leaves as a documented gap.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.policies.policy_engine import (
    AgentPermissionAlignment,
    AgentScopeEvaluation,
    ExternalActionEvaluation,
    PolicyEngine,
    PolicyLevel,
    WorkflowTriggerEvaluation,
)

from core.security.engine.security_decision import (
    SecurityDecisionPoint,
)


def _write_policy(tmp_dir: Path, *, risk_level: str, approval: str) -> Path:
    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": "research_agent",
                "resource": "web_search",
                "action": "search",
                "scope": "public_web",
                "risk_level": risk_level,
                "approval": approval,
            }
        ],
        "defaults": {
            "unknown_risk": "DENY",
            "unknown_permission": "DENY",
            "unknown_scope": "DENY",
            "authorization_failure": "DENY",
        },
    }
    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path


# ---------------------------------------------------------------------
# PolicyLevel
# ---------------------------------------------------------------------

def test_policy_level_matches_the_spec_declared_hierarchy_order():
    # POLICY_SPEC.md's Policy Hierarchy, in its own declared order --
    # asserted here so a future accidental reordering is caught.
    assert list(PolicyLevel) == [
        PolicyLevel.SYSTEM_CONSTITUTION,
        PolicyLevel.SECURITY_AND_SAFETY,
        PolicyLevel.HUMAN_APPROVAL,
        PolicyLevel.TOOL_RISK,
        PolicyLevel.AGENT_CONSTRAINTS,
        PolicyLevel.WORKFLOW_CONSTRAINTS,
    ]


def test_policy_level_lower_value_is_higher_priority():
    assert PolicyLevel.SYSTEM_CONSTITUTION < PolicyLevel.SECURITY_AND_SAFETY
    assert PolicyLevel.SECURITY_AND_SAFETY < PolicyLevel.HUMAN_APPROVAL
    assert PolicyLevel.HUMAN_APPROVAL < PolicyLevel.TOOL_RISK
    assert PolicyLevel.TOOL_RISK < PolicyLevel.AGENT_CONSTRAINTS
    assert PolicyLevel.AGENT_CONSTRAINTS < PolicyLevel.WORKFLOW_CONSTRAINTS


# ---------------------------------------------------------------------
# is_recovery_authorized (Failure Policy step 4)
# ---------------------------------------------------------------------

def test_is_recovery_authorized_true_only_for_decision_or_execution_error():
    engine = PolicyEngine()

    assert engine.is_recovery_authorized("DECISION_ERROR") is True
    assert engine.is_recovery_authorized("EXECUTION_ERROR") is True

    # Deliberate outcomes the loop reported on purpose -- never
    # authorized for a retry.
    assert engine.is_recovery_authorized("FAILED") is False
    assert engine.is_recovery_authorized("TOOL_ERROR") is False
    assert engine.is_recovery_authorized("APPROVAL_REQUIRED") is False
    assert engine.is_recovery_authorized("MAX_STEPS_EXCEEDED") is False
    assert engine.is_recovery_authorized("INVALID_ACTION") is False
    assert engine.is_recovery_authorized("COMPLETED") is False


# ---------------------------------------------------------------------
# evaluate_external_action (External Actions six-question checklist)
# ---------------------------------------------------------------------

def test_evaluate_external_action_answers_all_six_questions_for_an_allowed_low_risk_action():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir, risk_level="LOW", approval="none"
        )
        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )

        security_decision = security.evaluate(
            subject="research_agent",
            resource="web_search",
            action="search",
            scope="public_web",
        )

        engine = PolicyEngine()

        evaluation = engine.evaluate_external_action(
            action="search",
            subject="research_agent",
            tool_id="web_search",
            security_decision=security_decision,
        )

        assert evaluation == ExternalActionEvaluation(
            action="search",
            subject="research_agent",
            tool_id="web_search",
            risk_level="LOW",
            approval_required=False,
            verification_required=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_external_action_reports_approval_required_for_a_high_risk_action():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir, risk_level="HIGH", approval="policy"
        )
        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )

        security_decision = security.evaluate(
            subject="research_agent",
            resource="web_search",
            action="search",
            scope="public_web",
        )

        engine = PolicyEngine()

        evaluation = engine.evaluate_external_action(
            action="search",
            subject="research_agent",
            tool_id="web_search",
            security_decision=security_decision,
        )

        assert evaluation.risk_level == "HIGH"
        assert evaluation.approval_required is True
        assert evaluation.verification_required is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_external_action_rejects_empty_action():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir, risk_level="LOW", approval="none"
        )
        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )
        security_decision = security.evaluate(
            subject="research_agent",
            resource="web_search",
            action="search",
            scope="public_web",
        )

        engine = PolicyEngine()

        with pytest.raises(ValueError, match="action must be"):
            engine.evaluate_external_action(
                action="   ",
                subject="research_agent",
                tool_id="web_search",
                security_decision=security_decision,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_external_action_rejects_empty_subject():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir, risk_level="LOW", approval="none"
        )
        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )
        security_decision = security.evaluate(
            subject="research_agent",
            resource="web_search",
            action="search",
            scope="public_web",
        )

        engine = PolicyEngine()

        with pytest.raises(ValueError, match="subject must be"):
            engine.evaluate_external_action(
                action="search",
                subject="",
                tool_id="web_search",
                security_decision=security_decision,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_external_action_rejects_empty_tool_id():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir, risk_level="LOW", approval="none"
        )
        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )
        security_decision = security.evaluate(
            subject="research_agent",
            resource="web_search",
            action="search",
            scope="public_web",
        )

        engine = PolicyEngine()

        with pytest.raises(ValueError, match="tool_id must be"):
            engine.evaluate_external_action(
                action="search",
                subject="research_agent",
                tool_id="",
                security_decision=security_decision,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_external_action_rejects_a_security_decision_missing_effective_risk():
    # A lightweight duck-typed stand-in, not a real SecurityDecision --
    # exercising this completeness check does not require constructing
    # a full, real SecurityDecision object graph (which cannot itself
    # be missing this field; all its own fields are required at
    # construction). See evaluate_external_action's own docstring for
    # why this method is duck-typed.
    engine = PolicyEngine()

    stand_in = SimpleNamespace(
        authorization=SimpleNamespace(effective_risk=None),
        approval=SimpleNamespace(required=False),
    )

    with pytest.raises(ValueError, match="authorization.effective_risk"):
        engine.evaluate_external_action(
            action="search",
            subject="research_agent",
            tool_id="web_search",
            security_decision=stand_in,
        )


def test_evaluate_external_action_rejects_a_security_decision_missing_approval_required():
    engine = PolicyEngine()

    stand_in = SimpleNamespace(
        authorization=SimpleNamespace(effective_risk="LOW"),
        approval=SimpleNamespace(required=None),
    )

    with pytest.raises(ValueError, match="approval.required"):
        engine.evaluate_external_action(
            action="search",
            subject="research_agent",
            tool_id="web_search",
            security_decision=stand_in,
        )


def test_evaluate_external_action_rejects_a_security_decision_missing_entirely():
    # Passing something with no `.authorization`/`.approval` attributes
    # at all (e.g. a caller error) fails the same completeness check,
    # with the same clear error -- no separate type-check branch is
    # needed.
    engine = PolicyEngine()

    with pytest.raises(ValueError, match="authorization.effective_risk"):
        engine.evaluate_external_action(
            action="search",
            subject="research_agent",
            tool_id="web_search",
            security_decision="not a real SecurityDecision",
        )


# ---------------------------------------------------------------------
# evaluate_agent_scope (Agent Constraints -- declared tool-id scope,
# Build Phase 9). See core/policies/policy_engine.py's own module
# docstring and core/agents/research_agent.py / writer_agent.py's own
# docstrings for what this check does and does not cover.
# ---------------------------------------------------------------------

def test_evaluate_agent_scope_reports_within_scope_when_actual_tools_match_declared():
    engine = PolicyEngine()

    evaluation = engine.evaluate_agent_scope(
        subject="research_agent",
        declared_tool_ids={"web_search", "read_document"},
        actual_tool_ids={"web_search", "read_document"},
    )

    assert evaluation == AgentScopeEvaluation(
        subject="research_agent",
        declared_tool_ids=frozenset({"web_search", "read_document"}),
        actual_tool_ids=frozenset({"web_search", "read_document"}),
        unauthorized_tool_ids=frozenset(),
        within_scope=True,
    )


def test_evaluate_agent_scope_reports_within_scope_when_actual_tools_are_a_strict_subset_of_declared():
    # Declaring a tool never requires actually registering it -- only
    # registering something UNDECLARED is a scope violation.
    engine = PolicyEngine()

    evaluation = engine.evaluate_agent_scope(
        subject="research_agent",
        declared_tool_ids={"web_search", "read_document", "read_webpage"},
        actual_tool_ids={"web_search"},
    )

    assert evaluation.within_scope is True
    assert evaluation.unauthorized_tool_ids == frozenset()


def test_evaluate_agent_scope_reports_unauthorized_tools_when_actual_exceeds_declared():
    engine = PolicyEngine()

    evaluation = engine.evaluate_agent_scope(
        subject="research_agent",
        declared_tool_ids={"web_search"},
        actual_tool_ids={"web_search", "shell"},
    )

    assert evaluation.within_scope is False
    assert evaluation.unauthorized_tool_ids == frozenset({"shell"})


def test_evaluate_agent_scope_rejects_empty_subject():
    engine = PolicyEngine()

    with pytest.raises(ValueError, match="subject must be"):
        engine.evaluate_agent_scope(
            subject="   ",
            declared_tool_ids={"web_search"},
            actual_tool_ids={"web_search"},
        )


# ---------------------------------------------------------------------
# evaluate_agent_permission_alignment (Agent Constraints -- config-side
# alignment between registered tools and permissions.json's real
# grants, Build Phase 10). A second, complementary slice of "operate
# only within declared responsibilities" alongside evaluate_agent_scope
# above -- see that method's own docstring for the code-side check.
# ---------------------------------------------------------------------

def test_evaluate_agent_permission_alignment_reports_aligned_when_grants_exactly_match():
    engine = PolicyEngine()

    grants = {("web_search", "search", "public_web")}

    evaluation = engine.evaluate_agent_permission_alignment(
        subject="research_agent",
        tool_grants_needed=grants,
        security_grants_present=grants,
    )

    assert evaluation == AgentPermissionAlignment(
        subject="research_agent",
        tool_grants_needed=frozenset(grants),
        security_grants_present=frozenset(grants),
        missing_grants=frozenset(),
        extra_grants=frozenset(),
        aligned=True,
    )


def test_evaluate_agent_permission_alignment_reports_missing_grants_for_a_tool_with_no_permission():
    engine = PolicyEngine()

    evaluation = engine.evaluate_agent_permission_alignment(
        subject="research_agent",
        tool_grants_needed={("web_search", "search", "public_web")},
        security_grants_present=set(),
    )

    assert evaluation.aligned is False
    assert evaluation.missing_grants == frozenset(
        {("web_search", "search", "public_web")}
    )
    assert evaluation.extra_grants == frozenset()


def test_evaluate_agent_permission_alignment_reports_extra_grants_no_tool_needs():
    engine = PolicyEngine()

    evaluation = engine.evaluate_agent_permission_alignment(
        subject="research_agent",
        tool_grants_needed=set(),
        security_grants_present={("shell", "execute", "workspace")},
    )

    assert evaluation.aligned is False
    assert evaluation.missing_grants == frozenset()
    assert evaluation.extra_grants == frozenset(
        {("shell", "execute", "workspace")}
    )


def test_evaluate_agent_permission_alignment_rejects_empty_subject():
    engine = PolicyEngine()

    with pytest.raises(ValueError, match="subject must be"):
        engine.evaluate_agent_permission_alignment(
            subject="",
            tool_grants_needed=set(),
            security_grants_present=set(),
        )


# ---------------------------------------------------------------------
# evaluate_workflow_trigger() -- POLICY_SPEC.md's Workflow Constraints,
# v1 (Build Phase 12). See policy_engine.py's own module docstring
# (WORKFLOW CONSTRAINTS) for exactly what this one declared transition
# covers and does not.
# ---------------------------------------------------------------------

def test_evaluate_workflow_trigger_triggers_reviewer_agent_after_a_successful_write_report():
    engine = PolicyEngine()

    evaluation = engine.evaluate_workflow_trigger(
        completed_subject="writer_agent",
        tool_id="write_report",
        tool_status="SUCCESS",
    )

    assert evaluation == WorkflowTriggerEvaluation(
        completed_subject="writer_agent",
        tool_id="write_report",
        tool_status="SUCCESS",
        should_trigger=True,
        next_subject="reviewer_agent",
    )


def test_evaluate_workflow_trigger_does_not_trigger_for_a_denied_write_report():
    """
    Only a SUCCESSful write_report is a real, published report worth
    independently verifying -- a DENIED/APPROVAL_REQUIRED/ERROR result
    never actually wrote anything, so there is nothing yet for
    reviewer_agent to review.
    """
    engine = PolicyEngine()

    evaluation = engine.evaluate_workflow_trigger(
        completed_subject="writer_agent",
        tool_id="write_report",
        tool_status="DENIED",
    )

    assert evaluation.should_trigger is False
    assert evaluation.next_subject is None


def test_evaluate_workflow_trigger_does_not_trigger_for_an_unrelated_subject_or_tool():
    """
    research_agent's own write_research_findings is a different
    (subject, tool_id) pair entirely -- no transition is declared for
    it (see this project's own reasoning, in policy_engine.py's module
    docstring, for why research_agent -> writer_agent is deliberately
    not auto-triggered).
    """
    engine = PolicyEngine()

    evaluation = engine.evaluate_workflow_trigger(
        completed_subject="research_agent",
        tool_id="write_research_findings",
        tool_status="SUCCESS",
    )

    assert evaluation.should_trigger is False
    assert evaluation.next_subject is None


def test_evaluate_workflow_trigger_never_raises_when_tool_id_or_status_are_none():
    """
    A completed run that never invoked a tool at all has no tool_id/
    tool_status to report -- this must degrade to "no transition",
    never raise, mirroring the duck-typed tolerance
    evaluate_external_action()/Kernel._evaluate_policy already
    establish for similarly incomplete data.
    """
    engine = PolicyEngine()

    evaluation = engine.evaluate_workflow_trigger(
        completed_subject="writer_agent",
        tool_id=None,
        tool_status=None,
    )

    assert evaluation.should_trigger is False
    assert evaluation.next_subject is None


def test_evaluate_workflow_trigger_rejects_empty_completed_subject():
    engine = PolicyEngine()

    with pytest.raises(ValueError, match="completed_subject must be"):
        engine.evaluate_workflow_trigger(
            completed_subject="",
            tool_id="write_report",
            tool_status="SUCCESS",
        )

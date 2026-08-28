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
    AgentScopeEvaluation,
    ExternalActionEvaluation,
    PolicyEngine,
    PolicyLevel,
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

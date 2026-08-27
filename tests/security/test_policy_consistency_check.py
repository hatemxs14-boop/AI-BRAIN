"""
Regression tests for AuthorizationEngine's documented-defaults check.

`permissions.json` can optionally include `defaults` and `risk_levels`
sections that *document* the fail-closed/approval behavior
AuthorizationEngine and ApprovalGate actually enforce in Python code.
Those sections were never read at decision time -- editing them had
zero effect on real authorization behavior, which is misleading
config: SECURITY_SPEC.md's own principles call for "complete and
inspectable authorization decisions". AuthorizationEngine now verifies
at load time that these sections, when present, still match what the
code actually does, and fails loudly (a load-time `ValueError`, not a
silent no-op) if they've drifted.

This must be an opt-in check, not new required boilerplate: a policy
file that omits `defaults`/`risk_levels` entirely (like the project's
existing test fixtures throughout this suite) must continue to load
without complaint.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from core.security.engine.authorization import AuthorizationEngine


def _write_policy(tmp_dir: Path, policy: dict) -> Path:
    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path


def test_policy_without_documented_defaults_loads_cleanly():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir,
            {
                "version": "1.0",
                "permissions": [],
            },
        )

        # Must not raise.
        AuthorizationEngine(str(policy_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_policy_with_matching_documented_defaults_loads_cleanly():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir,
            {
                "version": "1.0",
                "permissions": [],
                "defaults": {
                    "unknown_risk": "DENY",
                    "unknown_permission": "DENY",
                    "unknown_scope": "DENY",
                    "authorization_failure": "DENY",
                },
                "risk_levels": {
                    "LOW": {"approval": "none", "default_decision": "ALLOW"},
                    "MEDIUM": {
                        "approval": "none",
                        "default_decision": "ALLOW_WITH_CONTROLS",
                    },
                    "HIGH": {
                        "approval": "policy",
                        "default_decision": "REQUIRE_APPROVAL",
                    },
                    "CRITICAL": {
                        "approval": "human",
                        "default_decision": "REQUIRE_APPROVAL",
                    },
                },
            },
        )

        # Must not raise -- this is exactly the real
        # core/security/schemas/permissions.json shape.
        AuthorizationEngine(str(policy_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_policy_with_stale_defaults_fails_loudly_at_load_time():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir,
            {
                "version": "1.0",
                "permissions": [],
                "defaults": {
                    "unknown_risk": "ALLOW",  # drifted from the code
                    "unknown_permission": "DENY",
                    "unknown_scope": "DENY",
                    "authorization_failure": "DENY",
                },
            },
        )

        with pytest.raises(ValueError, match="defaults"):
            AuthorizationEngine(str(policy_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_policy_with_stale_risk_levels_fails_loudly_at_load_time():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy_path = _write_policy(
            tmp_dir,
            {
                "version": "1.0",
                "permissions": [],
                "risk_levels": {
                    "LOW": {"approval": "none", "default_decision": "ALLOW"},
                    "MEDIUM": {
                        "approval": "human",  # drifted from the code
                        "default_decision": "ALLOW_WITH_CONTROLS",
                    },
                    "HIGH": {
                        "approval": "policy",
                        "default_decision": "REQUIRE_APPROVAL",
                    },
                    "CRITICAL": {
                        "approval": "human",
                        "default_decision": "REQUIRE_APPROVAL",
                    },
                },
            },
        )

        with pytest.raises(ValueError, match="risk_levels"):
            AuthorizationEngine(str(policy_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_real_project_policy_file_passes_the_consistency_check():
    """
    The actual, checked-in permissions.json must itself pass this
    check -- otherwise every real SecurityDecisionPoint construction
    in the whole test suite would start failing.
    """

    # Must not raise.
    AuthorizationEngine("core/security/schemas/permissions.json")

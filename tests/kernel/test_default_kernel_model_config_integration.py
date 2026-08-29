"""
Tests for build_default_kernel()'s `model_config_path` parameter
(Build Phase 18, core/kernel/default_kernel.py + core/llm/model_config.py).

Only the wiring is exercised here -- that `model_config_path` is
purely additive (an explicit llm_client_factory/decision_engine_factory
always wins), that it fills in model/temperature/max_tokens only where
the caller left them at None, and that the existing "exactly one of
llm_client_factory/decision_engine_factory" requirement still holds
when neither it nor model_config_path is given. The model config
file's own validation is tested in tests/llm/test_model_config.py --
this file never needs a real vendor SDK, since every scenario here
either supplies its own decision_engine_factory/llm_client_factory
(bypassing real client construction) or only checks that
build_default_kernel() read the file and set `model`/`temperature`/
`max_tokens` correctly, without ever calling the resulting factory.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.llm_decision_engine import LLMDecisionEngine

from core.kernel.default_kernel import build_default_kernel

from core.llm.llm_client import LLMClient
from core.llm.llm_response import LLMResponse


class _ImmediateCompleteEngine(AgentDecisionEngine):
    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Nothing to do.",
        )


class _StubLLMClient(LLMClient):
    def generate(self, request):
        return LLMResponse(content="unused", model="stub")


def _make_roots():
    docs_root = tempfile.mkdtemp()
    findings_root = tempfile.mkdtemp()
    audit_dir = tempfile.mkdtemp()
    return docs_root, findings_root, audit_dir


def _cleanup(*dirs):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def _write_model_config(tmp_dir: Path, content: dict) -> Path:
    path = tmp_dir / "model_config.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_model_config_path_none_by_default_changes_nothing():
    """
    Every existing caller passes no model_config_path -- confirm the
    default (None) still requires an explicit llm_client_factory or
    decision_engine_factory, exactly as before this Build Phase.
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        with pytest.raises(
            ValueError,
            match="llm_client_factory or decision_engine_factory",
        ):
            build_default_kernel(
                documents_root=docs_root,
                findings_root=findings_root,
                serper_api_key="test-key",
                audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            )
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_explicit_decision_engine_factory_wins_over_model_config_path():
    """
    An explicit decision_engine_factory must win even when a
    (deliberately invalid, to prove it is never even read)
    model_config_path is also supplied.
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=_ImmediateCompleteEngine,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            model_config_path="/this/path/does/not/exist/at/all.json",
        )

        assert [r.subject for r in kernel._registrations] == [
            "research_agent",
            "writer_agent",
            "reviewer_agent",
        ]
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_explicit_llm_client_factory_wins_over_model_config_path():
    """
    An explicit llm_client_factory must also win over
    model_config_path -- again proven by pointing model_config_path at
    a file that does not exist and confirming no error is raised.
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        kernel = build_default_kernel(
            llm_client_factory=_StubLLMClient,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            model_config_path="/this/path/does/not/exist/at/all.json",
        )

        assert [r.subject for r in kernel._registrations] == [
            "research_agent",
            "writer_agent",
            "reviewer_agent",
        ]
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_invalid_model_config_path_raises_when_no_factory_given():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        with pytest.raises(Exception, match="not found or unreadable"):
            build_default_kernel(
                documents_root=docs_root,
                findings_root=findings_root,
                serper_api_key="test-key",
                audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
                model_config_path="/this/path/does/not/exist/at/all.json",
            )
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_model_config_path_fills_in_model_and_max_tokens_when_omitted():
    """
    When only model_config_path is given, its own model/temperature/
    max_tokens must reach the LLMDecisionEngine each of the three
    registered agents will build_decision_engine() through.

    build_decision_engine() constructs LLMDecisionEngine(client, ...)
    eagerly, which means it calls llm_client_factory() -- i.e. actually
    builds a real ClaudeProvider(anthropic.Anthropic(...)) -- so this
    test needs the real 'anthropic' package installed (it is not, in
    this sandbox; skip-guarded exactly like core/llm/model_config.py's
    own "builds a real provider" tests in tests/llm/test_model_config.py,
    and for the same reason). A real (but never actually used to call
    any API -- constructing the SDK client performs no network call)
    API key value is enough; only the wiring is under test here.
    """
    pytest.importorskip("anthropic")

    tmp_dir = Path(tempfile.mkdtemp())
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        os.environ["SOME_VAR_NEVER_SET_XYZ"] = "sk-fake-but-present"

        config_path = _write_model_config(
            tmp_dir,
            {
                "provider": "anthropic",
                "model": "claude-config-model",
                "api_key_env": "SOME_VAR_NEVER_SET_XYZ",
                "max_tokens": 777,
            },
        )

        kernel = build_default_kernel(
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            model_config_path=config_path,
        )

        registration = kernel._registrations[0]
        decision_engine = registration.build_decision_engine()

        assert isinstance(decision_engine, LLMDecisionEngine)
        assert decision_engine.model == "claude-config-model"
        assert decision_engine.max_tokens == 777
        assert decision_engine.temperature is None
    finally:
        os.environ.pop("SOME_VAR_NEVER_SET_XYZ", None)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _cleanup(docs_root, findings_root, audit_dir)


def test_explicit_model_wins_over_model_config_path_value():
    """
    A caller-supplied `model=` argument must win over the value in the
    model config file -- the file only fills in what the caller left
    at None.

    Same real-'anthropic'-package requirement, and for the same
    reason, as test_model_config_path_fills_in_model_and_max_tokens_
    when_omitted above.
    """
    pytest.importorskip("anthropic")

    tmp_dir = Path(tempfile.mkdtemp())
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        os.environ["SOME_VAR_NEVER_SET_XYZ"] = "sk-fake-but-present"

        config_path = _write_model_config(
            tmp_dir,
            {
                "provider": "anthropic",
                "model": "claude-config-model",
                "api_key_env": "SOME_VAR_NEVER_SET_XYZ",
            },
        )

        kernel = build_default_kernel(
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            model_config_path=config_path,
            model="caller-explicit-model",
        )

        registration = kernel._registrations[0]
        decision_engine = registration.build_decision_engine()

        assert decision_engine.model == "caller-explicit-model"
    finally:
        os.environ.pop("SOME_VAR_NEVER_SET_XYZ", None)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _cleanup(docs_root, findings_root, audit_dir)

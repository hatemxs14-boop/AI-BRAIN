"""
Tests for build_default_kernel()'s new `guardrail_engine`/
`token_budget`/`model_tier_router`/`semantic_embedding_client`
parameters (Build Phase 30, core/kernel/default_kernel.py).

Context: for four Build Phases (23, 26, 27, 28) these components were
real, tested, and working at the Kernel class level, but
build_default_kernel() -- the one convenience function this whole
project uses to assemble a production Kernel -- had no parameter for
any of them at all, so the only way to actually use one was to bypass
this function and construct Kernel() by hand. This file proves the fix:
each of these four now reaches the underlying Kernel exactly, with the
same "every existing caller's behavior, cost, and test counts are
unchanged unless it explicitly opts in" guarantee every other optional
parameter on this function already provides (see
test_default_kernel_response_cache_integration.py's own identical
convention for `enable_response_cache`).

Two kinds of proof, per the pattern this project already established:
identity-level wiring checks (the object build_default_kernel() was
given is the exact object the returned Kernel now holds), plus at
least one genuine end-to-end behavioral proof (token_budget actually
triggers a real BUDGET_EXCEEDED result through a real
build_default_kernel()-built Kernel's own run()) -- not just "the
attribute got set."
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from core.agents.guardrails import OutputGuardrailEngine
from core.llm.embeddings import EmbeddingClient
from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.model_tier import ModelTierRouter
from core.llm.token_usage import TokenUsage

from core.kernel.default_kernel import build_default_kernel

from core.llm.budget import TokenBudget


_COMPLETE_RESPONSE_TEXT = (
    '{"action_type":"COMPLETE","tool_id":null,"inputs":null,'
    '"reason":"Done."}'
)


class _StubLLMClient(LLMClient):
    def __init__(self, *, usage: TokenUsage | None = None):
        self._usage = usage

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=_COMPLETE_RESPONSE_TEXT,
            model=request.model or "stub",
            usage=self._usage,
        )


class _FakeEmbeddingClient(EmbeddingClient):
    def embed(self, texts, *, input_type):
        return tuple((1.0, 0.0) for _ in texts)


def _make_roots():
    docs_root = tempfile.mkdtemp()
    findings_root = tempfile.mkdtemp()
    audit_dir = tempfile.mkdtemp()
    return docs_root, findings_root, audit_dir


def _cleanup(*dirs):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------
# Defaults -- every existing caller is unaffected
# ---------------------------------------------------------------------


def test_all_four_default_to_none():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        assert kernel.guardrail_engine is None
        assert kernel.token_budget is None
        assert kernel.model_tier_router is None
        assert kernel.semantic_embedding_client is None
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


# ---------------------------------------------------------------------
# Identity-level wiring -- the exact object passed in reaches the
# underlying Kernel, unmodified.
# ---------------------------------------------------------------------


def test_guardrail_engine_reaches_the_underlying_kernel():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        engine = OutputGuardrailEngine(enforce=True)

        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            guardrail_engine=engine,
        )

        assert kernel.guardrail_engine is engine
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_token_budget_reaches_the_underlying_kernel():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        budget = TokenBudget(max_total_tokens=100)

        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            token_budget=budget,
        )

        assert kernel.token_budget is budget
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_model_tier_router_reaches_the_underlying_kernel():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        router = ModelTierRouter(
            simple_model="cheap-model", complex_model="expensive-model"
        )

        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            model_tier_router=router,
        )

        assert kernel.model_tier_router is router
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_semantic_embedding_client_reaches_the_underlying_kernel():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        client = _FakeEmbeddingClient()

        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            semantic_embedding_client=client,
        )

        assert kernel.semantic_embedding_client is client
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


# ---------------------------------------------------------------------
# Genuine end-to-end behavioral proof, not just attribute wiring
# ---------------------------------------------------------------------


def test_token_budget_actually_triggers_budget_exceeded_end_to_end():
    """
    The strongest possible proof this wiring is real: a
    build_default_kernel()-built Kernel, with a real TokenBudget
    passed straight through, must actually produce a real
    BUDGET_EXCEEDED result on a real (fake-LLM-backed) run() call --
    not just carry the object as an inert attribute.
    """

    docs_root, findings_root, audit_dir = _make_roots()
    try:
        usage = TokenUsage(
            prompt_tokens=80, completion_tokens=50, total_tokens=130
        )

        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(usage=usage),
            temperature=0.0,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            token_budget=TokenBudget(max_total_tokens=100),
        )

        result = kernel.run("Research AI agents")

        assert result.status == "BUDGET_EXCEEDED"
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_guardrail_engine_actually_blocks_end_to_end_when_enforcing():
    """
    Same genuine-delegation proof, for `guardrail_engine`: a real,
    enforcing OutputGuardrailEngine passed through
    build_default_kernel() must actually block a real HIGH-severity
    action (a credential-shaped value in the decision's own text) on a
    real run() call.
    """

    docs_root, findings_root, audit_dir = _make_roots()
    try:
        leaking_response = LLMResponse(
            content=(
                '{"action_type":"COMPLETE","tool_id":null,"inputs":null,'
                '"reason":"Here is the key: sk-abcdefghijklmnopqrstuvwx"}'
            ),
            model="stub",
        )

        class _LeakingClient(LLMClient):
            def generate(self, request: LLMRequest) -> LLMResponse:
                return leaking_response

        kernel = build_default_kernel(
            llm_client_factory=lambda: _LeakingClient(),
            temperature=0.0,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            guardrail_engine=OutputGuardrailEngine(enforce=True),
        )

        result = kernel.run("Research AI agents")

        assert result.status == "GUARDRAIL_BLOCKED"
    finally:
        _cleanup(docs_root, findings_root, audit_dir)

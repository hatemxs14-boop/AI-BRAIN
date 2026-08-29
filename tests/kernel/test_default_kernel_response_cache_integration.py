"""
Tests for build_default_kernel()'s `enable_response_cache`/
`response_cache_max_entries`/`response_cache_nondeterministic`
parameters (Build Phase 20, core/kernel/default_kernel.py +
core/llm/caching_llm_client.py).

Only the wiring is exercised here -- that `enable_response_cache`
defaults to off (zero behavior change for every existing caller), that
when enabled the SAME ResponseCache is genuinely shared across
separate `build_decision_engine()` calls (simulating separate
Kernel.run() attempts against the same Kernel instance), and that
`response_cache_nondeterministic` threads through to CachingLLMClient
correctly. CachingLLMClient's own caching rules are tested directly in
tests/llm/test_caching_llm_client.py -- this file never needs a real
vendor SDK, since every scenario here supplies its own
`llm_client_factory` (a real, in-process test double), bypassing real
client construction entirely.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from core.agents.agent_context import AgentContext
from core.agents.llm_decision_engine import LLMDecisionEngine

from core.kernel.default_kernel import build_default_kernel

from core.llm.caching_llm_client import CachingLLMClient
from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse


_COMPLETE_RESPONSE_TEXT = (
    '{"action_type":"COMPLETE","tool_id":null,"inputs":null,'
    '"reason":"Done."}'
)


class _CountingLLMClient(LLMClient):
    def __init__(self):
        self.call_count = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=_COMPLETE_RESPONSE_TEXT, model="stub")


def _make_roots():
    docs_root = tempfile.mkdtemp()
    findings_root = tempfile.mkdtemp()
    audit_dir = tempfile.mkdtemp()
    return docs_root, findings_root, audit_dir


def _cleanup(*dirs):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def test_response_cache_disabled_by_default():
    """
    Every existing caller passes no `enable_response_cache` -- confirm
    the default (False) means two identical decisions across two
    separately-built decision engines both genuinely call the wrapped
    client, exactly as before this Build Phase.
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        counting_client = _CountingLLMClient()

        kernel = build_default_kernel(
            llm_client_factory=lambda: counting_client,
            temperature=0.0,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        registration = kernel._registrations[0]

        engine1 = registration.build_decision_engine()
        engine1.decide(AgentContext(task="Do the same thing."))

        engine2 = registration.build_decision_engine()
        engine2.decide(AgentContext(task="Do the same thing."))

        assert isinstance(engine1.client, _CountingLLMClient)
        assert counting_client.call_count == 2
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_response_cache_shares_hits_across_separate_decision_engine_builds():
    """
    With `enable_response_cache=True`, two separately-built decision
    engines (simulating two separate Kernel.run() attempts against the
    same Kernel instance -- e.g. a RECOVER IF NEEDED retry, or two
    calls sharing an identical opening prompt) must share ONE
    ResponseCache: the second, structurally-identical decide() call
    must be served from cache and never reach the wrapped client.
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        counting_client = _CountingLLMClient()

        kernel = build_default_kernel(
            llm_client_factory=lambda: counting_client,
            temperature=0.0,
            enable_response_cache=True,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        registration = kernel._registrations[0]

        engine1 = registration.build_decision_engine()
        assert isinstance(engine1.client, CachingLLMClient)
        engine1.decide(AgentContext(task="Do the same thing."))

        # A fresh LLMDecisionEngine AND a fresh CachingLLMClient
        # instance -- but backed by the same shared ResponseCache.
        engine2 = registration.build_decision_engine()
        assert isinstance(engine2.client, CachingLLMClient)
        assert engine2.client is not engine1.client
        engine2.decide(AgentContext(task="Do the same thing."))

        assert counting_client.call_count == 1
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_response_cache_does_not_cache_nondeterministic_requests_by_default():
    """
    Without an explicit `temperature=0.0` (the default temperature=None
    reaches LLMRequest as None, which CachingLLMClient treats as
    non-deterministic by default -- see its own docstring), enabling
    the cache must still not produce a hit.
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        counting_client = _CountingLLMClient()

        kernel = build_default_kernel(
            llm_client_factory=lambda: counting_client,
            enable_response_cache=True,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        registration = kernel._registrations[0]

        engine1 = registration.build_decision_engine()
        engine1.decide(AgentContext(task="Do the same thing."))

        engine2 = registration.build_decision_engine()
        engine2.decide(AgentContext(task="Do the same thing."))

        assert counting_client.call_count == 2
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_response_cache_nondeterministic_flag_threads_through():
    """
    `response_cache_nondeterministic=True` must actually reach
    CachingLLMClient's own `cache_nondeterministic` -- proven the same
    genuine-delegation way this project already proves other flags
    reach their target (e.g. Build Phase 18's context_retrieval_limit
    test): a request that would NOT be cached by default (temperature
    left at None) IS cached once this flag is set.
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        counting_client = _CountingLLMClient()

        kernel = build_default_kernel(
            llm_client_factory=lambda: counting_client,
            enable_response_cache=True,
            response_cache_nondeterministic=True,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        registration = kernel._registrations[0]

        engine1 = registration.build_decision_engine()
        assert engine1.client.cache_nondeterministic is True
        engine1.decide(AgentContext(task="Do the same thing."))

        engine2 = registration.build_decision_engine()
        engine2.decide(AgentContext(task="Do the same thing."))

        assert counting_client.call_count == 1
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_response_cache_max_entries_threads_through():
    """
    `response_cache_max_entries` must actually reach the shared
    ResponseCache's own bound -- proven by filling it past a
    deliberately tiny limit and confirming the earliest entry was
    evicted (a real cache miss on a request that would otherwise still
    have been a hit).
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        counting_client = _CountingLLMClient()

        kernel = build_default_kernel(
            llm_client_factory=lambda: counting_client,
            temperature=0.0,
            enable_response_cache=True,
            response_cache_max_entries=1,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        registration = kernel._registrations[0]

        engine1 = registration.build_decision_engine()
        engine1.decide(AgentContext(task="First distinct task."))

        engine2 = registration.build_decision_engine()
        engine2.decide(AgentContext(task="Second distinct task."))  # evicts the first

        engine3 = registration.build_decision_engine()
        engine3.decide(AgentContext(task="First distinct task."))  # cache miss again

        assert counting_client.call_count == 3
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_response_cache_not_applied_when_decision_engine_factory_is_explicit():
    """
    `enable_response_cache` must have no effect when the caller
    supplies its own `decision_engine_factory` directly -- there is no
    `llm_client_factory` for it to wrap in that case, exactly mirroring
    `model_config_path`'s own "an explicit decision_engine_factory
    always wins" precedent (Build Phase 18).
    """
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        counting_client = _CountingLLMClient()

        def build_engine():
            return LLMDecisionEngine(counting_client, temperature=0.0)

        kernel = build_default_kernel(
            decision_engine_factory=build_engine,
            enable_response_cache=True,
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        registration = kernel._registrations[0]
        engine = registration.build_decision_engine()

        assert isinstance(engine.client, _CountingLLMClient)
    finally:
        _cleanup(docs_root, findings_root, audit_dir)

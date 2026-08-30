"""
Tests for core.observability.langfuse_trace (Build Phase 32):
TraceRecorder's own ABC contract, LangfuseTraceRecorder's input
validation and correct delegation to a fake, in-process stand-in for a
real `langfuse.Langfuse` client (never the real network),
build_langfuse_trace_recorder_factory()'s three real, actionable error
paths (empty constructor args, missing environment variables, missing
vendor package) plus its happy path when `langfuse` genuinely is
installed.

Same two-tier honesty split tests/llm/test_embeddings.py and
tests/evaluation/test_output_quality.py already document: the
missing-package path is verified for real in ANY environment via
`monkeypatch.setitem(sys.modules, "langfuse", None)`, while the
"builds a real LangfuseTraceRecorder when langfuse IS installed" path
is skip-guarded with `pytest.importorskip` and only runs for real on a
machine that has it installed -- this sandbox does not (confirmed: no
PyPI access here), so that one test is expected to report "skipped"
in this sandbox's own `pytest -v -rs` output.

One deliberate difference from tests/evaluation/test_output_quality.py's
own missing-package test: Build Phase 31's fix there was needed
because `deepeval` ships its own pytest plugin (registered via
`entry_points`) that pre-loads `sys.modules["deepeval.metrics"]` etc.
*before* any test runs, so patching only the top-level `"deepeval"`
key was not enough once code did dotted-submodule imports like
`from deepeval.metrics import GEval` (Python resolves an
already-cached dotted submodule directly, without re-checking the
parent's `None` sentinel). core.observability.langfuse_trace's own
factory never does that -- it only ever executes a single, flat
`import langfuse` statement (see its own module docstring and
build_langfuse_trace_recorder_factory()'s `factory()` body) and then
attribute-accesses `langfuse.Langfuse`. For a flat `import langfuse`,
CPython's import system checks `sys.modules["langfuse"]` directly; if
that value is `None`, it raises ImportError immediately, regardless of
whether some *other* dotted name like `langfuse.something` happens to
already be cached (which would only matter for a dotted import of that
specific submodule, never attempted here). So a single-key
`monkeypatch.setitem(sys.modules, "langfuse", None)` is sufficient and
correct here even if `langfuse` also ships its own pytest plugin --
unlike deepeval's case, this module's import shape never touches a
cached submodule path that could bypass the sentinel.
"""
from __future__ import annotations

import os
import sys

import pytest

from core.llm.token_usage import TokenUsage
from core.observability.langfuse_trace import (
    LangfuseConfigError,
    LangfuseTraceRecorder,
    TraceRecorder,
    build_langfuse_trace_recorder_factory,
)


# ---------------------------------------------------------------------
# TraceRecorder -- ABC contract
# ---------------------------------------------------------------------


def test_trace_recorder_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        TraceRecorder()


def test_trace_recorder_flush_defaults_to_a_no_op():
    class _MinimalRecorder(TraceRecorder):
        def record_run(
            self,
            *,
            name,
            input_text,
            output_text,
            status,
            metadata=None,
            usage=None,
        ):
            pass

    # Must not raise -- the base class's own flush() is a concrete,
    # non-abstract no-op, deliberately never auto-called by
    # record_run() itself (see the class's own docstring for why).
    _MinimalRecorder().flush()


# ---------------------------------------------------------------------
# LangfuseTraceRecorder -- construction and a fake vendor client
# ---------------------------------------------------------------------


class _FakeObservation:
    def __init__(self, calls):
        self._calls = calls

    def update(self, **kwargs):
        self._calls.append(("update", kwargs))


class _FakeObservationContext:
    def __init__(self, calls, start_kwargs):
        self._calls = calls
        self._start_kwargs = start_kwargs

    def __enter__(self):
        return _FakeObservation(self._calls)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeLangfuseClient:
    def __init__(self):
        self.calls = []
        self.flush_calls = 0

    def start_as_current_observation(self, **kwargs):
        self.calls.append(("start_as_current_observation", kwargs))
        return _FakeObservationContext(self.calls, kwargs)

    def flush(self):
        self.flush_calls += 1


def test_langfuse_trace_recorder_rejects_empty_name():
    recorder = LangfuseTraceRecorder(_FakeLangfuseClient())

    with pytest.raises(ValueError, match="name"):
        recorder.record_run(
            name="   ",
            input_text="hello",
            output_text="world",
            status="COMPLETED",
        )


def test_langfuse_trace_recorder_rejects_empty_input_text():
    recorder = LangfuseTraceRecorder(_FakeLangfuseClient())

    with pytest.raises(ValueError, match="input_text"):
        recorder.record_run(
            name="kernel_run",
            input_text="",
            output_text="world",
            status="COMPLETED",
        )


def test_langfuse_trace_recorder_rejects_a_non_string_output_text():
    recorder = LangfuseTraceRecorder(_FakeLangfuseClient())

    with pytest.raises(TypeError, match="output_text"):
        recorder.record_run(
            name="kernel_run",
            input_text="hello",
            output_text=123,
            status="COMPLETED",
        )


def test_langfuse_trace_recorder_rejects_empty_status():
    recorder = LangfuseTraceRecorder(_FakeLangfuseClient())

    with pytest.raises(ValueError, match="status"):
        recorder.record_run(
            name="kernel_run",
            input_text="hello",
            output_text="world",
            status="",
        )


def test_langfuse_trace_recorder_rejects_a_non_mapping_metadata():
    recorder = LangfuseTraceRecorder(_FakeLangfuseClient())

    with pytest.raises(TypeError, match="metadata"):
        recorder.record_run(
            name="kernel_run",
            input_text="hello",
            output_text="world",
            status="COMPLETED",
            metadata=["not", "a", "mapping"],
        )


def test_langfuse_trace_recorder_rejects_a_non_token_usage_usage():
    recorder = LangfuseTraceRecorder(_FakeLangfuseClient())

    with pytest.raises(TypeError, match="usage"):
        recorder.record_run(
            name="kernel_run",
            input_text="hello",
            output_text="world",
            status="COMPLETED",
            usage={"prompt_tokens": 1},
        )


def test_langfuse_trace_recorder_starts_a_generation_observation():
    client = _FakeLangfuseClient()
    recorder = LangfuseTraceRecorder(client)

    recorder.record_run(
        name="kernel_run",
        input_text="Research AI agents",
        output_text="Done.",
        status="COMPLETED",
    )

    start_calls = [
        c for c in client.calls if c[0] == "start_as_current_observation"
    ]
    assert len(start_calls) == 1
    call_kwargs = start_calls[0][1]
    assert call_kwargs["as_type"] == "generation"
    assert call_kwargs["name"] == "kernel_run"


def test_langfuse_trace_recorder_updates_the_observation_with_input_and_output():
    client = _FakeLangfuseClient()
    recorder = LangfuseTraceRecorder(client)

    recorder.record_run(
        name="kernel_run",
        input_text="Research AI agents",
        output_text="Done.",
        status="COMPLETED",
    )

    update_calls = [c for c in client.calls if c[0] == "update"]
    assert len(update_calls) == 1
    update_kwargs = update_calls[0][1]
    assert update_kwargs["input"] == "Research AI agents"
    assert update_kwargs["output"] == "Done."
    assert update_kwargs["usage"] is None


def test_langfuse_trace_recorder_merges_status_into_metadata():
    client = _FakeLangfuseClient()
    recorder = LangfuseTraceRecorder(client)

    recorder.record_run(
        name="kernel_run",
        input_text="Research AI agents",
        output_text="Done.",
        status="COMPLETED",
        metadata={"subject": "writer_agent", "recovery_attempts": 0},
    )

    update_kwargs = [c for c in client.calls if c[0] == "update"][0][1]
    assert update_kwargs["metadata"] == {
        "subject": "writer_agent",
        "recovery_attempts": 0,
        "status": "COMPLETED",
    }


def test_langfuse_trace_recorder_builds_status_only_metadata_when_none_given():
    client = _FakeLangfuseClient()
    recorder = LangfuseTraceRecorder(client)

    recorder.record_run(
        name="kernel_run",
        input_text="Research AI agents",
        output_text="Done.",
        status="FAILED",
    )

    update_kwargs = [c for c in client.calls if c[0] == "update"][0][1]
    assert update_kwargs["metadata"] == {"status": "FAILED"}


def test_langfuse_trace_recorder_translates_token_usage_to_the_usage_payload_shape():
    client = _FakeLangfuseClient()
    recorder = LangfuseTraceRecorder(client)

    usage = TokenUsage(
        prompt_tokens=12, completion_tokens=8, total_tokens=20
    )

    recorder.record_run(
        name="kernel_run",
        input_text="Research AI agents",
        output_text="Done.",
        status="COMPLETED",
        usage=usage,
    )

    update_kwargs = [c for c in client.calls if c[0] == "update"][0][1]
    assert update_kwargs["usage"] == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    }


def test_langfuse_trace_recorder_flush_delegates_to_the_client():
    client = _FakeLangfuseClient()
    recorder = LangfuseTraceRecorder(client)

    recorder.flush()
    recorder.flush()

    assert client.flush_calls == 2


# ---------------------------------------------------------------------
# build_langfuse_trace_recorder_factory() -- construction validation
# ---------------------------------------------------------------------


def test_factory_rejects_empty_public_key_env():
    with pytest.raises(LangfuseConfigError, match="public_key_env"):
        build_langfuse_trace_recorder_factory(
            public_key_env="   ",
            secret_key_env="LANGFUSE_SECRET_KEY",
            base_url="http://localhost:3000",
        )


def test_factory_rejects_empty_secret_key_env():
    with pytest.raises(LangfuseConfigError, match="secret_key_env"):
        build_langfuse_trace_recorder_factory(
            public_key_env="LANGFUSE_PUBLIC_KEY",
            secret_key_env="   ",
            base_url="http://localhost:3000",
        )


def test_factory_rejects_empty_base_url():
    with pytest.raises(LangfuseConfigError, match="base_url"):
        build_langfuse_trace_recorder_factory(
            public_key_env="LANGFUSE_PUBLIC_KEY",
            secret_key_env="LANGFUSE_SECRET_KEY",
            base_url="",
        )


def test_factory_returns_callable_without_side_effects():
    # Building the factory must not touch the environment or import
    # `langfuse` -- only *calling* the returned factory does.
    factory = build_langfuse_trace_recorder_factory(
        public_key_env="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
        secret_key_env="SOME_OTHER_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
        base_url="http://localhost:3000",
    )

    assert callable(factory)


# ---------------------------------------------------------------------
# build_langfuse_trace_recorder_factory() -- calling the factory
# ---------------------------------------------------------------------


def test_factory_missing_public_key_env_raises_before_any_sdk_import():
    os.environ.pop("SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ", None)

    factory = build_langfuse_trace_recorder_factory(
        public_key_env="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
        secret_key_env="SOME_OTHER_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
        base_url="http://localhost:3000",
    )

    with pytest.raises(
        LangfuseConfigError,
        match="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
    ):
        factory()


def test_factory_missing_secret_key_env_raises():
    os.environ["FAKE_LANGFUSE_PUBLIC_KEY_XYZ"] = "pk-fake"
    os.environ.pop("SOME_MISSING_LANGFUSE_SECRET_XYZ", None)
    try:
        factory = build_langfuse_trace_recorder_factory(
            public_key_env="FAKE_LANGFUSE_PUBLIC_KEY_XYZ",
            secret_key_env="SOME_MISSING_LANGFUSE_SECRET_XYZ",
            base_url="http://localhost:3000",
        )

        with pytest.raises(
            LangfuseConfigError,
            match="SOME_MISSING_LANGFUSE_SECRET_XYZ",
        ):
            factory()
    finally:
        os.environ.pop("FAKE_LANGFUSE_PUBLIC_KEY_XYZ", None)


def test_factory_empty_secret_key_env_raises():
    os.environ["FAKE_LANGFUSE_PUBLIC_KEY_XYZ2"] = "pk-fake"
    os.environ["FAKE_LANGFUSE_EMPTY_SECRET_XYZ"] = ""
    try:
        factory = build_langfuse_trace_recorder_factory(
            public_key_env="FAKE_LANGFUSE_PUBLIC_KEY_XYZ2",
            secret_key_env="FAKE_LANGFUSE_EMPTY_SECRET_XYZ",
            base_url="http://localhost:3000",
        )

        with pytest.raises(
            LangfuseConfigError, match="FAKE_LANGFUSE_EMPTY_SECRET_XYZ"
        ):
            factory()
    finally:
        os.environ.pop("FAKE_LANGFUSE_PUBLIC_KEY_XYZ2", None)
        os.environ.pop("FAKE_LANGFUSE_EMPTY_SECRET_XYZ", None)


def test_factory_missing_package_raises_clear_error(monkeypatch):
    # Simulates "langfuse is not installed" deterministically, in ANY
    # environment -- see this module's own top-of-file docstring for
    # why a single-key monkeypatch is correct here (unlike Build Phase
    # 31's deepeval fix, this factory only ever does a flat
    # `import langfuse`, never a dotted-submodule import).
    monkeypatch.setitem(sys.modules, "langfuse", None)

    os.environ["FAKE_LANGFUSE_PUBLIC_KEY_XYZ3"] = "pk-fake"
    os.environ["FAKE_LANGFUSE_SECRET_KEY_XYZ3"] = "sk-fake"
    try:
        factory = build_langfuse_trace_recorder_factory(
            public_key_env="FAKE_LANGFUSE_PUBLIC_KEY_XYZ3",
            secret_key_env="FAKE_LANGFUSE_SECRET_KEY_XYZ3",
            base_url="http://localhost:3000",
        )

        with pytest.raises(
            LangfuseConfigError, match="langfuse.*not installed"
        ):
            factory()
    finally:
        os.environ.pop("FAKE_LANGFUSE_PUBLIC_KEY_XYZ3", None)
        os.environ.pop("FAKE_LANGFUSE_SECRET_KEY_XYZ3", None)


def test_factory_builds_real_langfuse_trace_recorder_when_installed():
    pytest.importorskip("langfuse")

    os.environ["REAL_LANGFUSE_PUBLIC_KEY_XYZ"] = "pk-fake-but-present"
    os.environ["REAL_LANGFUSE_SECRET_KEY_XYZ"] = "sk-fake-but-present"
    try:
        factory = build_langfuse_trace_recorder_factory(
            public_key_env="REAL_LANGFUSE_PUBLIC_KEY_XYZ",
            secret_key_env="REAL_LANGFUSE_SECRET_KEY_XYZ",
            base_url="http://localhost:3000",
        )

        recorder = factory()

        assert isinstance(recorder, LangfuseTraceRecorder)
    finally:
        os.environ.pop("REAL_LANGFUSE_PUBLIC_KEY_XYZ", None)
        os.environ.pop("REAL_LANGFUSE_SECRET_KEY_XYZ", None)

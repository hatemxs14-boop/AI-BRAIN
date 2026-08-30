"""
Tests for core.evaluation.output_quality (Build Phase 31): self-hosted
DeepEval-based output-quality evaluation via Ollama.

Everything that does not require the `deepeval` package installed runs
for real, unconditionally, in this sandbox: OutputQualityResult's own
validation, the JudgeModel ABC contract, OllamaJudgeModel's real
request-building/response-parsing logic against a fake HTTP layer
(`http_post=`, exactly mirroring tests/agents/test_llama_guard.py's own
identical convention for Ollama), and OutputQualityEvaluator's own
constructor validation (which runs entirely before any deepeval
import). The "deepeval is not installed" error path is exercised both
because it genuinely is not installed here, AND forced deterministically
by patching every submodule this project's own import statements
actually touch (`deepeval`, `deepeval.metrics`, `deepeval.models`,
`deepeval.test_case`) to `None` in `sys.modules`.

Real-machine correction, from the user's first real run of this suite
with `deepeval` genuinely installed: patching only the top-level
`"deepeval"` key (the same technique tests/llm/test_embeddings.py uses
for `voyageai`) is NOT sufficient here, because `deepeval` ships its
own pytest plugin, auto-loaded by pytest itself via `entry_points`
before any test runs -- this pre-populates `sys.modules
["deepeval.metrics"]` (etc.) with real modules ahead of time, and
Python's import machinery resolves a dotted submodule directly from
its own cached entry once present, never re-checking the parent's
`None` sentinel. Every submodule actually imported from must be
patched individually for the ImportError to be forced reliably. The
real GEval measurement happy path (`pytest.importorskip("deepeval")`-
guarded) is now CONFIRMED passing for real on the user's machine --
see core/evaluation/output_quality.py's own top-of-file docstring for
the full account of what that run proved.
"""
from __future__ import annotations

import sys

import pytest
import requests

from core.evaluation.output_quality import (
    DEFAULT_JUDGE_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_BASE_URL,
    JudgeModel,
    JudgeModelError,
    OllamaJudgeModel,
    OutputQualityEvaluationError,
    OutputQualityEvaluator,
    OutputQualityResult,
)


class _FakeResponse:

    def __init__(self, *, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload configured")
        return self._payload


class _FakeJudgeModel(JudgeModel):

    def __init__(self, *, name="fake-judge", response="looks good"):
        self._name = name
        self._response = response
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


# ---------------------------------------------------------------------
# OutputQualityResult validation
# ---------------------------------------------------------------------


def test_output_quality_result_accepts_valid_data():
    result = OutputQualityResult(
        score=0.75, passed=True, reason="good enough", metric_name="Quality"
    )
    assert result.score == 0.75
    assert result.passed is True
    assert result.reason == "good enough"
    assert result.metric_name == "Quality"


def test_output_quality_result_allows_empty_reason():
    result = OutputQualityResult(
        score=0.0, passed=False, reason="", metric_name="Quality"
    )
    assert result.reason == ""


def test_output_quality_result_rejects_non_numeric_score():
    with pytest.raises(TypeError, match="score"):
        OutputQualityResult(
            score="high", passed=True, reason="", metric_name="Quality"
        )


def test_output_quality_result_rejects_bool_score():
    with pytest.raises(TypeError, match="score"):
        OutputQualityResult(
            score=True, passed=True, reason="", metric_name="Quality"
        )


def test_output_quality_result_rejects_out_of_range_score():
    with pytest.raises(ValueError, match="0.0, 1.0"):
        OutputQualityResult(
            score=1.5, passed=True, reason="", metric_name="Quality"
        )


def test_output_quality_result_rejects_non_bool_passed():
    with pytest.raises(TypeError, match="passed"):
        OutputQualityResult(
            score=0.5, passed="yes", reason="", metric_name="Quality"
        )


def test_output_quality_result_rejects_non_string_reason():
    with pytest.raises(TypeError, match="reason"):
        OutputQualityResult(
            score=0.5, passed=True, reason=123, metric_name="Quality"
        )


def test_output_quality_result_rejects_empty_metric_name():
    with pytest.raises(TypeError, match="metric_name"):
        OutputQualityResult(
            score=0.5, passed=True, reason="", metric_name="   "
        )


# ---------------------------------------------------------------------
# JudgeModel -- ABC contract
# ---------------------------------------------------------------------


def test_judge_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        JudgeModel()


# ---------------------------------------------------------------------
# OllamaJudgeModel -- construction validation
# ---------------------------------------------------------------------


def test_ollama_judge_model_requires_model():
    with pytest.raises(ValueError, match="model"):
        OllamaJudgeModel(model="")


def test_ollama_judge_model_uses_documented_defaults():
    judge = OllamaJudgeModel(model="llama3.1")
    assert judge.base_url == DEFAULT_OLLAMA_BASE_URL
    assert judge.timeout == DEFAULT_JUDGE_TIMEOUT_SECONDS
    assert judge.model_name == "llama3.1"


def test_ollama_judge_model_strips_trailing_slash_from_base_url():
    judge = OllamaJudgeModel(
        model="llama3.1", base_url="http://localhost:11434/"
    )
    assert judge.base_url == "http://localhost:11434"


def test_ollama_judge_model_rejects_empty_base_url():
    with pytest.raises(ValueError, match="base_url"):
        OllamaJudgeModel(model="llama3.1", base_url="   ")


def test_ollama_judge_model_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout"):
        OllamaJudgeModel(model="llama3.1", timeout=0)


def test_ollama_judge_model_rejects_bool_timeout():
    with pytest.raises(ValueError, match="timeout"):
        OllamaJudgeModel(model="llama3.1", timeout=True)


# ---------------------------------------------------------------------
# OllamaJudgeModel.generate() -- input validation
# ---------------------------------------------------------------------


def test_generate_rejects_empty_prompt():
    judge = OllamaJudgeModel(
        model="llama3.1",
        http_post=lambda *a, **k: _FakeResponse(payload={"response": "ok"}),
    )
    with pytest.raises(ValueError, match="prompt"):
        judge.generate("   ")


# ---------------------------------------------------------------------
# OllamaJudgeModel.generate() -- real request building
# ---------------------------------------------------------------------


def test_generate_posts_to_the_generate_endpoint_with_expected_payload():
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(payload={"response": "This output is great."})

    judge = OllamaJudgeModel(
        model="llama3.1",
        base_url="http://localhost:11434",
        timeout=45.0,
        http_post=fake_post,
    )

    output = judge.generate("Rate this output.")

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["kwargs"]["json"] == {
        "model": "llama3.1",
        "prompt": "Rate this output.",
        "stream": False,
    }
    assert captured["kwargs"]["timeout"] == 45.0
    assert output == "This output is great."


# ---------------------------------------------------------------------
# OllamaJudgeModel.generate() -- error paths
# ---------------------------------------------------------------------


def test_generate_wraps_request_exceptions():
    def failing_post(*a, **k):
        raise requests.ConnectionError("connection refused")

    judge = OllamaJudgeModel(model="llama3.1", http_post=failing_post)

    with pytest.raises(JudgeModelError, match="request .* failed"):
        judge.generate("hello")


def test_generate_raises_on_non_200_status():
    judge = OllamaJudgeModel(
        model="llama3.1",
        http_post=lambda *a, **k: _FakeResponse(
            status_code=500, text="internal error"
        ),
    )

    with pytest.raises(JudgeModelError, match="status=500"):
        judge.generate("hello")


def test_generate_raises_on_invalid_json():
    judge = OllamaJudgeModel(
        model="llama3.1",
        http_post=lambda *a, **k: _FakeResponse(payload=None),
    )

    with pytest.raises(JudgeModelError, match="not valid JSON"):
        judge.generate("hello")


def test_generate_raises_on_non_dict_json_payload():
    judge = OllamaJudgeModel(
        model="llama3.1",
        http_post=lambda *a, **k: _FakeResponse(payload=["not", "a", "dict"]),
    )

    with pytest.raises(JudgeModelError, match="unexpected response shape"):
        judge.generate("hello")


def test_generate_raises_when_response_field_is_missing():
    judge = OllamaJudgeModel(
        model="llama3.1",
        http_post=lambda *a, **k: _FakeResponse(payload={}),
    )

    with pytest.raises(JudgeModelError, match="no usable 'response' field"):
        judge.generate("hello")


def test_generate_raises_when_response_field_is_blank():
    judge = OllamaJudgeModel(
        model="llama3.1",
        http_post=lambda *a, **k: _FakeResponse(payload={"response": "   "}),
    )

    with pytest.raises(JudgeModelError, match="no usable 'response' field"):
        judge.generate("hello")


# ---------------------------------------------------------------------
# OutputQualityEvaluator -- constructor validation (runs entirely
# before any deepeval import, so all of this is real regardless of
# whether deepeval is installed)
# ---------------------------------------------------------------------


def test_evaluator_rejects_non_judge_model():
    with pytest.raises(TypeError, match="judge_model"):
        OutputQualityEvaluator(
            judge_model="not-a-judge-model",
            name="Quality",
            criteria="Is it good?",
        )


def test_evaluator_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):
        OutputQualityEvaluator(
            judge_model=_FakeJudgeModel(),
            name="   ",
            criteria="Is it good?",
        )


def test_evaluator_rejects_neither_criteria_nor_steps():
    with pytest.raises(ValueError, match="exactly one"):
        OutputQualityEvaluator(
            judge_model=_FakeJudgeModel(), name="Quality"
        )


def test_evaluator_rejects_both_criteria_and_steps():
    with pytest.raises(ValueError, match="not both"):
        OutputQualityEvaluator(
            judge_model=_FakeJudgeModel(),
            name="Quality",
            criteria="Is it good?",
            evaluation_steps=["Check completeness."],
        )


def test_evaluator_rejects_empty_evaluation_steps():
    with pytest.raises(ValueError, match="exactly one"):
        OutputQualityEvaluator(
            judge_model=_FakeJudgeModel(),
            name="Quality",
            evaluation_steps=[],
        )


def test_evaluator_rejects_out_of_range_threshold():
    with pytest.raises(ValueError, match="threshold"):
        OutputQualityEvaluator(
            judge_model=_FakeJudgeModel(),
            name="Quality",
            criteria="Is it good?",
            threshold=1.5,
        )


def test_evaluator_rejects_bool_threshold():
    with pytest.raises(ValueError, match="threshold"):
        OutputQualityEvaluator(
            judge_model=_FakeJudgeModel(),
            name="Quality",
            criteria="Is it good?",
            threshold=True,
        )


def test_evaluator_accepts_criteria_only():
    evaluator = OutputQualityEvaluator(
        judge_model=_FakeJudgeModel(),
        name="Quality",
        criteria="Is the output complete and relevant?",
    )
    assert evaluator.criteria == "Is the output complete and relevant?"
    assert evaluator.evaluation_steps is None


def test_evaluator_accepts_evaluation_steps_only():
    evaluator = OutputQualityEvaluator(
        judge_model=_FakeJudgeModel(),
        name="Quality",
        evaluation_steps=["Check completeness.", "Check relevance."],
    )
    assert evaluator.criteria is None
    assert evaluator.evaluation_steps == (
        "Check completeness.",
        "Check relevance.",
    )


# ---------------------------------------------------------------------
# OutputQualityEvaluator.evaluate() -- input validation (also runs
# before any deepeval import)
# ---------------------------------------------------------------------


def test_evaluate_rejects_empty_input_text():
    evaluator = OutputQualityEvaluator(
        judge_model=_FakeJudgeModel(), name="Quality", criteria="Good?"
    )
    with pytest.raises(ValueError, match="input_text"):
        evaluator.evaluate(input_text="   ", actual_output="hi")


def test_evaluate_rejects_empty_actual_output():
    evaluator = OutputQualityEvaluator(
        judge_model=_FakeJudgeModel(), name="Quality", criteria="Good?"
    )
    with pytest.raises(ValueError, match="actual_output"):
        evaluator.evaluate(input_text="task", actual_output="   ")


def test_evaluate_rejects_blank_expected_output():
    evaluator = OutputQualityEvaluator(
        judge_model=_FakeJudgeModel(), name="Quality", criteria="Good?"
    )
    with pytest.raises(ValueError, match="expected_output"):
        evaluator.evaluate(
            input_text="task", actual_output="hi", expected_output="   "
        )


def test_evaluate_rejects_blank_context():
    evaluator = OutputQualityEvaluator(
        judge_model=_FakeJudgeModel(), name="Quality", criteria="Good?"
    )
    with pytest.raises(ValueError, match="context"):
        evaluator.evaluate(
            input_text="task", actual_output="hi", context="   "
        )


# ---------------------------------------------------------------------
# OutputQualityEvaluator.evaluate() -- "deepeval is not installed"
# ---------------------------------------------------------------------


def test_evaluate_raises_clear_error_when_deepeval_not_installed(
    monkeypatch,
):
    # Deterministic in ANY environment -- documented CPython import
    # behavior, identical technique to tests/llm/test_embeddings.py's
    # own `voyageai` equivalent, with one real-machine-confirmed
    # refinement: unlike `voyageai`, `deepeval` ships its own pytest
    # plugin, which pytest auto-loads via entry_points at startup --
    # this means `sys.modules["deepeval.metrics"]` (etc.) are already
    # populated as REAL modules before this test ever runs whenever
    # deepeval is genuinely installed. Patching only the top-level
    # "deepeval" key is therefore not enough: `from deepeval.metrics
    # import GEval` resolves "deepeval.metrics" directly against
    # sys.modules under its own exact dotted name and never re-checks
    # the "deepeval" parent entry once that submodule is already
    # cached, so the None sentinel on "deepeval" alone is silently
    # bypassed. Every submodule this module actually imports from must
    # be patched to None individually for the ImportError to be
    # forced, regardless of what pytest's own plugin loading already
    # cached.
    for module_name in (
        "deepeval",
        "deepeval.metrics",
        "deepeval.models",
        "deepeval.test_case",
    ):
        monkeypatch.setitem(sys.modules, module_name, None)

    evaluator = OutputQualityEvaluator(
        judge_model=_FakeJudgeModel(), name="Quality", criteria="Good?"
    )

    with pytest.raises(
        OutputQualityEvaluationError, match="deepeval.*not installed"
    ):
        evaluator.evaluate(input_text="task", actual_output="output")


def test_evaluate_without_deepeval_installed_at_all():
    # Checks whether the *package* is actually installed (not merely
    # whether some earlier test has already imported it into
    # sys.modules) -- this sandbox genuinely has no deepeval installed
    # (no PyPI access), so this proves the same path for real here,
    # with no monkeypatching at all.
    import importlib.util

    if importlib.util.find_spec("deepeval") is not None:
        pytest.skip("deepeval is installed in this environment")

    evaluator = OutputQualityEvaluator(
        judge_model=_FakeJudgeModel(), name="Quality", criteria="Good?"
    )

    with pytest.raises(
        OutputQualityEvaluationError, match="deepeval.*not installed"
    ):
        evaluator.evaluate(input_text="task", actual_output="output")


# ---------------------------------------------------------------------
# OutputQualityEvaluator.evaluate() -- real GEval measurement, only
# when deepeval is genuinely installed (never run in this sandbox)
# ---------------------------------------------------------------------


def test_evaluate_produces_a_real_score_when_deepeval_is_installed():
    pytest.importorskip("deepeval")

    judge = _FakeJudgeModel(
        response=(
            '{"score": 0.9, "reason": "The output directly and '
            'completely answers the question."}'
        )
    )

    evaluator = OutputQualityEvaluator(
        judge_model=judge,
        name="Correctness",
        evaluation_steps=[
            "Check whether the output directly answers the input.",
            "Penalize incomplete or off-topic answers.",
        ],
    )

    result = evaluator.evaluate(
        input_text="What is 2+2?",
        actual_output="4",
        expected_output="4",
    )

    assert isinstance(result, OutputQualityResult)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.passed, bool)
    assert result.metric_name == "Correctness"
    assert judge.calls  # the judge model was genuinely invoked

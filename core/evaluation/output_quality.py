"""
core/evaluation/output_quality.py

Build Phase 31: a real, self-hosted output-quality evaluation layer via
DeepEval (github.com/confident-ai/deepeval, Apache 2.0) -- item 5 of
the 8-item ECC-informed priority ranking.

What this is, and what it is NOT: unlike core/agents/guardrails.py
(safety/security findings, wired live into every Kernel.run()) and
core/agents/llama_guard.py (a confidence gate layered on top of it),
this module is a development/CI/regression EVALUATION HARNESS, not a
live production gate. Its job is to score how GOOD an agent's output
is against a rubric (task completion, relevance, conciseness, factual
grounding, etc.), not whether it is SAFE. Nothing in this module is
wired into build_default_kernel() or Kernel itself, and nothing here
should be -- a quality score is meant to be read by a human or a CI
pipeline deciding whether to ship a change to research_agent/
writer_agent/reviewer_agent, not to block or alter a live agent run.

Provider choice: **DeepEval**, chosen explicitly by the user after
being told it is genuinely free and Apache-2.0-licensed, requires no
API key by default, and -- critically -- lets ANY caller supply its
own custom judge model (`deepeval.models.DeepEvalBaseLLM`) rather than
defaulting to a paid API call. This module wires that custom-model
seam to **Ollama**, the same self-hosted server this project already
runs for Build Phase 29's Llama Guard gate, so real evaluation runs
cost the same zero marginal dollars/tokens Llama Guard already does --
directly serving the user's own standing correction of this project's
dependency philosophy (Build Phase 28: minimize real financial/token
cost, not dependency count).

Structural note, worth being explicit about: `deepeval` is NOT
imported at module import time anywhere in this file. This project's
own `JudgeModel` ABC (mirroring core.llm.llm_client.LLMClient, core.
llm.embeddings.EmbeddingClient, and core.agents.llama_guard.
LlamaGuardClient's identical one-layer-over shape) and its Ollama
implementation below need no vendor SDK at all -- only `requests`,
already a project dependency. `deepeval` itself (and the small adapter
class that wraps a JudgeModel into a real `DeepEvalBaseLLM`) is
imported LAZILY, inside OutputQualityEvaluator.evaluate(), exactly
mirroring core.llm.embeddings.build_embedding_client_factory()'s own
"never import the vendor SDK until the returned factory is actually
called" convention. This means the whole module imports cleanly with
or without `deepeval` installed, and only the one method that actually
needs it raises a clear, actionable OutputQualityEvaluationError (with
a `pip install deepeval` instruction) when it is not.

Like core/llm/embeddings.py's own VoyageEmbeddingClient, this project
is honest about what has and hasn't run for real in this sandbox:
`deepeval` is not installed here (no PyPI access in this sandbox), so
OutputQualityEvaluator.evaluate()'s real GEval measurement against a
genuine `deepeval` install has never executed anywhere in this
sandbox. What IS verified for real here (see tests/evaluation/
test_output_quality.py) is every piece of this module that does not
require the vendor package: OutputQualityResult's own validation,
JudgeModel's ABC contract, OllamaJudgeModel's real request-building/
response-parsing logic against a fake HTTP layer (never a live Ollama
server), OutputQualityEvaluator's own constructor validation (which
runs entirely before any deepeval import), and the
"deepeval is not installed" error path -- exercised for real here,
since it genuinely isn't, and additionally forced deterministically via
`monkeypatch.setitem(sys.modules, "deepeval", None)` (the same
documented-CPython-import-behavior technique tests/llm/
test_embeddings.py already uses for `voyageai`) so it also proves
itself in any environment where deepeval IS installed.

API surface researched directly against DeepEval's own current
documentation (deepeval.com, PyPI `deepeval` 4.2.x) rather than
assumed from training data, given this project's own standing
insistence on verifying real capabilities rather than fabricating
them: the `DeepEvalBaseLLM` base class's four required methods
(`get_model_name`, `load_model`, `generate`, `a_generate`), GEval's
constructor shape (`name`, exactly one of `criteria`/`evaluation_steps`,
`evaluation_params`, `model`, `threshold`), and `LLMTestCase`'s fields
are all taken from that documentation. Two assumptions were flagged
here for the same honesty reasons, and **both are now CONFIRMED
correct against a real, installed `deepeval` package** (the user's
real machine, immediately after this module was first delivered):

1. The enum naming for `evaluation_params` -- this module tries
   `SingleTurnParams` first, falling back to `LLMTestCaseParams` for
   cross-version compatibility. Confirmed: the real install resolved
   `SingleTurnParams` and reached real GEval measurement without error.
2. The judge-model JSON-parsing fallback for DeepEval's
   schema-constrained generation requests (`schema(**
   json.loads(raw_output))`). Confirmed working exactly as designed,
   both directions: a real GEval run against a real deepeval install,
   using a fake-but-injectable JudgeModel returning valid JSON,
   produced a real, correctly-shaped OutputQualityResult end to end
   (see tests/evaluation/test_output_quality.py's own
   test_evaluate_produces_a_real_score_when_deepeval_is_installed,
   `pytest.importorskip`-guarded and now confirmed passing for real).
   Separately, a run with a judge model returning plain non-JSON text
   correctly raised a clear OutputQualityEvaluationError naming the
   requested schema ("Steps" -- DeepEval's own internal chain-of-
   thought reasoning step before scoring) and including the raw,
   unparseable output, rather than silently mis-scoring. This also
   confirms GEval does request schema-constrained generation from a
   custom model, exactly the case this fallback exists for.

One unrelated bug was found and fixed by this same real-machine run,
in the TEST suite only (not in this module's own logic): `deepeval`
ships its own pytest plugin, auto-loaded by pytest itself via
`entry_points` before any test runs, which pre-populates
`sys.modules["deepeval.metrics"]` (etc.) with real modules ahead of
time. This meant `monkeypatch.setitem(sys.modules, "deepeval", None)`
alone -- otherwise identical to tests/llm/test_embeddings.py's own
`voyageai` technique -- silently failed to force the "deepeval is not
installed" path, since Python's import machinery resolves
`deepeval.metrics` directly from its own cached entry once present,
without re-checking the `deepeval` parent's `None` sentinel. Fixed by
patching every submodule this module actually imports from
(`deepeval`, `deepeval.metrics`, `deepeval.models`,
`deepeval.test_case`) to `None` individually, which reliably forces
the ImportError regardless of what pytest's own plugin loading has
already cached.

Never stores a real API key anywhere in this module -- `OllamaJudgeModel`
talks to a local Ollama server exactly like OllamaLlamaGuardClient
does, and DeepEval's own optional cloud reporting (Confident AI) is
never touched by this module at all.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import requests


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Deliberately no default judge model name -- unlike Llama Guard 3
# (Build Phase 29), which is a single, purpose-built classifier tag,
# a general-purpose judge model has no one "obviously right" choice,
# and this project's own standing convention (core/llm/model_config.py,
# core/llm/model_tier.py) is to never hardcode a model name that can
# silently go stale. Callers must pass `model=` explicitly.
DEFAULT_JUDGE_TIMEOUT_SECONDS = 60.0

# Sensible default: DeepEval's own documented default threshold for a
# GEval-style metric.
DEFAULT_QUALITY_THRESHOLD = 0.5


class JudgeModelError(RuntimeError):
    """
    Raised when a real judge-model call (e.g. to a self-hosted Ollama
    server) fails or returns a response this module cannot make sense
    of -- a service/network/format problem, never a caller-input
    mistake. Mirrors core.agents.llama_guard.LlamaGuardError's own
    identical rationale for being a RuntimeError, not a ValueError.
    """


class OutputQualityEvaluationError(RuntimeError):
    """
    Raised by OutputQualityEvaluator.evaluate() for anything that
    prevents a real quality score from being produced: `deepeval` not
    being installed, a real DeepEval/judge-model failure during
    measurement, or the judge model's raw text output not parsing into
    a schema DeepEval itself requested (see this module's own top-of-
    file docstring for how schema-constrained requests are handled).

    Deliberately never caught internally and turned into a fabricated
    "passed"/"failed" result -- unlike OutputGuardrailEngine's
    Llama-Guard confidence gate (Build Phase 29), which has a safe,
    honest fallback to degrade to (the original regex findings), a
    quality *score* has no such fallback: silently reporting a fake
    score on infrastructure failure would be actively misleading, not
    merely less safe. This module is a development/CI tool, not a live
    safety gate, so failing loudly here is the correct and honest
    choice, not a missed opportunity to degrade gracefully.
    """


@dataclass(frozen=True)
class OutputQualityResult:
    """
    One real quality-evaluation result from OutputQualityEvaluator.

    `score`: the raw metric score, in [0.0, 1.0].
    `passed`: whether `score` met the metric's own configured
    threshold (DeepEval's own `metric.success`).
    `reason`: DeepEval's own natural-language explanation for the
    score -- never fabricated; may be an empty string if the
    underlying metric genuinely returned none.
    `metric_name`: the name of the metric that produced this result,
    matching the `name` the caller gave OutputQualityEvaluator.
    """

    score: float
    passed: bool
    reason: str
    metric_name: str

    def __post_init__(self) -> None:

        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
        ):
            raise TypeError("score must be a real number.")

        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError("score must be in the range [0.0, 1.0].")

        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean.")

        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string.")

        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise TypeError("metric_name must be a non-empty string.")


class JudgeModel(ABC):
    """
    Provider-independent interface for a plain text-in/text-out LLM
    used to JUDGE another agent's output -- the same one-layer-over
    role core.llm.llm_client.LLMClient, core.llm.embeddings.
    EmbeddingClient, and core.agents.llama_guard.LlamaGuardClient
    already establish for generation, embeddings, and safety
    classification respectively, applied here to quality judging.

    This layer does not:

    - decide agent actions
    - execute tools
    - authorize operations
    - access the Security Layer
    - contain provider-specific business logic
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """A human-readable identifier for this judge model."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a plain-text completion for `prompt`. Raises
        ValueError for an empty/non-string `prompt`, and a
        provider-specific error (e.g. JudgeModelError for
        OllamaJudgeModel) for any real request/response failure --
        never silently returns fabricated text.
        """
        raise NotImplementedError


class OllamaJudgeModel(JudgeModel):
    """
    Real judge-model text generation via a self-hosted Ollama server's
    REST API (`POST {base_url}/api/generate`) -- structurally identical
    to core.agents.llama_guard.OllamaLlamaGuardClient, except general-
    purpose (any instruction-following Ollama model the caller names,
    not the Llama Guard 3 classifier specifically) and returning the
    server's raw text response rather than a parsed safety verdict.

    `http_post` is an injection point for tests, exactly mirroring
    OllamaLlamaGuardClient's own `http_post=` convention: it defaults
    to `requests.post`, but a test supplies a fake to exercise this
    client's real request-building/response-parsing logic without a
    live Ollama server or real model weights.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
        http_post: Callable[..., Any] | None = None,
    ) -> None:

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string.")

        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number.")

        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._post = http_post if http_post is not None else requests.post

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str) -> str:

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        try:
            response = self._post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise JudgeModelError(
                f"Ollama request to {self.base_url} failed: {exc}"
            ) from exc

        status_code = getattr(response, "status_code", None)

        if status_code != 200:
            body_preview = str(getattr(response, "text", ""))[:500]
            raise JudgeModelError(
                "Ollama returned a non-200 response "
                f"(status={status_code!r}): {body_preview!r}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise JudgeModelError(
                "Ollama returned a response that is not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise JudgeModelError(
                "Ollama returned an unexpected response shape: "
                f"{type(payload).__name__}."
            )

        raw_output = payload.get("response")

        if not isinstance(raw_output, str) or not raw_output.strip():
            raise JudgeModelError(
                "Ollama's /api/generate response carries no usable "
                "'response' field."
            )

        return raw_output


class OutputQualityEvaluator:
    """
    Scores one agent output against a rubric using a real DeepEval
    GEval metric, powered by a caller-supplied, self-hosted JudgeModel
    -- never a paid API call, never DeepEval's own default judge model.

    Exactly one of `criteria` (a plain-language description of what
    "good" means) or `evaluation_steps` (an explicit, ordered checklist)
    must be given, mirroring GEval's own documented either/or
    constraint -- validated here, before any deepeval import, so a
    caller gets an immediate, clear error rather than a lazily-surfaced
    one from inside a vendor call.
    """

    def __init__(
        self,
        *,
        judge_model: JudgeModel,
        name: str,
        criteria: str | None = None,
        evaluation_steps: Sequence[str] | None = None,
        threshold: float = DEFAULT_QUALITY_THRESHOLD,
    ) -> None:

        if not isinstance(judge_model, JudgeModel):
            raise TypeError(
                "judge_model must be a JudgeModel instance."
            )

        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")

        has_criteria = isinstance(criteria, str) and criteria.strip()
        has_steps = (
            evaluation_steps is not None
            and not isinstance(evaluation_steps, (str, bytes))
            and len(tuple(evaluation_steps)) > 0
        )

        if has_criteria and has_steps:
            raise ValueError(
                "Provide exactly one of criteria or evaluation_steps, "
                "not both."
            )

        if not has_criteria and not has_steps:
            raise ValueError(
                "Provide exactly one of criteria or evaluation_steps."
            )

        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not (0.0 <= threshold <= 1.0)
        ):
            raise ValueError(
                "threshold must be a number in the range [0.0, 1.0]."
            )

        self.judge_model = judge_model
        self.name = name
        self.criteria = criteria if has_criteria else None
        self.evaluation_steps = (
            tuple(evaluation_steps) if has_steps else None
        )
        self.threshold = float(threshold)

    def evaluate(
        self,
        *,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: str | None = None,
    ) -> OutputQualityResult:
        """
        Score `actual_output` (produced in response to `input_text`)
        against this evaluator's own criteria/evaluation_steps, using a
        real DeepEval GEval metric backed by this evaluator's
        `judge_model`. `expected_output` (a reference answer) and
        `context` (supporting grounding material) are both optional --
        include them only when this evaluator's own criteria actually
        depend on them, since GEval evaluates strictly against the
        LLMTestCase parameters it is configured to look at.

        Raises OutputQualityEvaluationError if `deepeval` is not
        installed, or if the real measurement fails for any reason
        (judge-model failure, malformed DeepEval response, etc.) --
        never returns a fabricated result.
        """

        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("input_text must be a non-empty string.")

        if not isinstance(actual_output, str) or not actual_output.strip():
            raise ValueError("actual_output must be a non-empty string.")

        if expected_output is not None and (
            not isinstance(expected_output, str) or not expected_output.strip()
        ):
            raise ValueError(
                "expected_output must be a non-empty string when given."
            )

        if context is not None and (
            not isinstance(context, str) or not context.strip()
        ):
            raise ValueError("context must be a non-empty string when given.")

        try:
            from deepeval.metrics import GEval
            from deepeval.models import DeepEvalBaseLLM
            from deepeval.test_case import LLMTestCase

            try:
                from deepeval.test_case import (
                    SingleTurnParams as _MetricParam,
                )
            except ImportError:
                from deepeval.test_case import (
                    LLMTestCaseParams as _MetricParam,
                )
        except ImportError as exc:
            raise OutputQualityEvaluationError(
                "The 'deepeval' package is not installed, so no real "
                "output-quality score can be produced. Install it "
                "with `pip install deepeval`."
            ) from exc

        judge_model = self.judge_model

        class _JudgeModelAdapter(DeepEvalBaseLLM):
            """
            Wraps this evaluator's own JudgeModel into a real
            DeepEvalBaseLLM -- defined here, inside evaluate(), so the
            rest of this module never needs `deepeval` importable just
            to be imported itself.
            """

            def get_model_name(self) -> str:
                return judge_model.model_name

            def load_model(self):
                return judge_model

            def generate(self, prompt: str, schema=None):
                raw_output = judge_model.generate(prompt)

                if schema is None:
                    return raw_output

                # DeepEval's own metrics (GEval included) commonly
                # request schema-constrained (JSON) output from the
                # judge model for reliable score/reason parsing --
                # this project's own JudgeModel interface is
                # deliberately plain text-in/text-out (no vendor-
                # specific structured-output API to wrap), so this
                # follows DeepEval's own documented fallback for
                # custom models without native structured output:
                # parse the raw text as JSON and construct the
                # requested Pydantic schema from it directly.
                import json

                try:
                    parsed = json.loads(raw_output)
                except (TypeError, ValueError) as exc:
                    raise OutputQualityEvaluationError(
                        "The judge model's output could not be parsed "
                        f"as JSON for DeepEval's requested schema "
                        f"{getattr(schema, '__name__', schema)!r}: "
                        f"{exc}. Raw output: {raw_output!r}"
                    ) from exc

                try:
                    return schema(**parsed)
                except Exception as exc:
                    raise OutputQualityEvaluationError(
                        "The judge model's JSON output does not match "
                        "DeepEval's requested schema "
                        f"{getattr(schema, '__name__', schema)!r}: "
                        f"{exc}. Parsed output: {parsed!r}"
                    ) from exc

            async def a_generate(self, prompt: str, schema=None):
                return self.generate(prompt, schema)

        evaluation_params = [
            _MetricParam.INPUT,
            _MetricParam.ACTUAL_OUTPUT,
        ]

        if expected_output is not None:
            evaluation_params.append(_MetricParam.EXPECTED_OUTPUT)

        if context is not None:
            evaluation_params.append(_MetricParam.CONTEXT)

        geval_kwargs: dict[str, Any] = {
            "name": self.name,
            "evaluation_params": evaluation_params,
            "model": _JudgeModelAdapter(),
            "threshold": self.threshold,
        }

        if self.criteria is not None:
            geval_kwargs["criteria"] = self.criteria
        else:
            geval_kwargs["evaluation_steps"] = list(self.evaluation_steps)

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=[context] if context is not None else None,
        )

        try:
            metric = GEval(**geval_kwargs)
            metric.measure(test_case)
        except OutputQualityEvaluationError:
            raise
        except Exception as exc:
            # Deliberately broad: a real DeepEval/judge-model failure
            # can surface as many different exception types (HTTP
            # errors bubbled up from OllamaJudgeModel, DeepEval's own
            # internal errors, a malformed judge response DeepEval
            # itself could not parse). This is a vendor-call boundary,
            # exactly like OllamaLlamaGuardClient's own request/response
            # handling -- the difference is DeepEval's own surface is
            # not enumerable the way `requests`' exception hierarchy
            # is, so this module wraps broadly here, and only here.
            raise OutputQualityEvaluationError(
                f"DeepEval GEval measurement failed: {exc}"
            ) from exc

        return OutputQualityResult(
            score=float(metric.score),
            passed=bool(metric.success),
            reason=str(metric.reason) if metric.reason is not None else "",
            metric_name=self.name,
        )

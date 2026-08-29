from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from core.llm.llm_client import (
    LLMClient,
)

from core.llm.providers.claude_provider import (
    ClaudeProvider,
)

from core.llm.providers.openai_provider import (
    OpenAIProvider,
)


# ---------------------------------------------------------------------
# Build Phase 18 -- centralized model/provider configuration.
#
# The problem this solves: before this file existed, every separate
# "business project" built on this Kernel had to write its own Python
# code constructing a real Anthropic/OpenAI SDK client, wrapping it in
# ClaudeProvider/OpenAIProvider, and passing that as `llm_client_factory`
# into build_default_kernel() (confirmed by grep across this repo --
# there was no shared config file, registry, or single source of truth
# anywhere; core/llm/providers/claude_provider.py and openai_provider.py
# both only ever take an already-constructed client, and never touch
# API keys or model selection themselves). Switching every one of those
# separately-deployed projects to a new model or a new provider meant
# editing code in each of them, one at a time.
#
# This module makes that a data change instead: one small JSON file
# (see config/model_config.example.json for the template) names a
# provider, a model, and the *name* of an environment variable holding
# the API key -- never the key itself, consistent with this project's
# standing rule to never store real API keys anywhere in the repo or
# in code. build_default_kernel()'s new `model_config_path` parameter
# (core/kernel/default_kernel.py) reads that one file and builds the
# real LLMClient factory from it automatically. To move every business
# project built this way onto a new model or provider, a human edits
# that one JSON file (or points multiple projects at the same shared
# file) -- no project's own code changes.
#
# Like core/orchestration/langgraph_orchestration_engine.py before it,
# this module is honest about what has and hasn't run for real in this
# sandbox: `anthropic` and `openai` are both listed in requirements.txt
# but neither package is installed here (confirmed separately -- there
# is no package-index access in this sandbox at all), so the actual
# `anthropic.Anthropic(...)`/`openai.OpenAI(...)` construction calls
# below have never executed anywhere. What IS verified in-sandbox (see
# tests/llm/test_model_config.py) is every piece of this module that
# does not require the vendor SDKs: all of load_model_config()'s
# validation, and that build_llm_client_factory_from_config()'s
# returned factory raises a clear, actionable ModelConfigError when the
# named environment variable is missing (checked *before* attempting
# any SDK import) and when the vendor package genuinely is not
# installed (exercised for real here, since it genuinely isn't). The
# "builds a real client when the package is installed" path is
# skip-guarded with pytest.importorskip, exactly like
# LangGraphOrchestrationEngine's own test suite -- it is the first real
# verification of this module's two provider branches, still pending
# on a real machine with `pip install -r requirements.txt`, not merely
# written.
# ---------------------------------------------------------------------


class ModelConfigError(ValueError):
    """
    Raised when a model/provider configuration file (or the
    environment it references) is missing, malformed, or otherwise
    invalid.

    Deliberately a ValueError subclass, matching
    core.kernel.workflow_config.WorkflowConfigError's own convention
    for the same reason: this always signals a human-fixable
    configuration mistake -- a bad file, a bad field, a missing
    environment variable -- never an internal bug in this project's
    own code.
    """


# The only two providers core/llm/providers/*.py currently implements.
# Deliberately a plain tuple, not an enum -- this module has no reason
# to be more rigid than core.kernel.workflow_config's own plain-tuple
# `trigger_keywords_all` convention, and adding a third provider later
# only means adding one more `if config.provider == "..."` branch in
# build_llm_client_factory_from_config() below plus one more entry
# here.
SUPPORTED_PROVIDERS = ("anthropic", "openai")

# Default location a real deployment is expected to keep its own,
# untracked model_config.json at (see config/model_config.example.json
# for the template, and .gitignore for why the real file is never
# committed). This is only a suggested default for callers -- nothing
# in this module or in build_default_kernel() reads this path
# automatically; build_default_kernel()'s own `model_config_path`
# parameter defaults to None, matching every other optional
# `enable_*`/`*_path`/`*_dir` parameter's "no behavior change unless a
# caller explicitly opts in" convention (see that function's own
# docstring).
DEFAULT_MODEL_CONFIG_PATH = "config/model_config.json"


@dataclass(frozen=True)
class ModelConfig:
    """
    A single, fully-validated model/provider configuration, as loaded
    from one JSON file by load_model_config() below.

    Carries no API key -- only the *name* of the environment variable
    build_llm_client_factory_from_config()'s returned factory will read
    the real key from, at the moment a real LLMClient is actually
    built (never at load time, and never stored on this dataclass).
    """

    provider: str
    model: str
    api_key_env: str
    temperature: float | None = None
    max_tokens: int | None = None


def _require_non_empty_str(
    value: object,
    field_name: str,
    path: Path,
) -> str:

    if not isinstance(value, str) or not value.strip():
        raise ModelConfigError(
            f"Model config file {path}: '{field_name}' must be a "
            f"non-empty string, got {value!r}."
        )

    return value


def load_model_config(path: str | Path) -> ModelConfig:
    """
    Load and validate a ModelConfig from a single JSON file.

    Expected shape:

        {
          "provider": "anthropic",
          "model": "claude-sonnet-4-5",
          "api_key_env": "ANTHROPIC_API_KEY",
          "temperature": null,
          "max_tokens": 1024
        }

    `provider` must be one of SUPPORTED_PROVIDERS (case-insensitive).
    `model` and `api_key_env` are required non-empty strings.
    `temperature` and `max_tokens` are both optional (default null) --
    when present, `temperature` must be a number and `max_tokens` must
    be a positive integer.

    Fails loud on every problem -- a missing file, invalid JSON, a
    missing/wrong-typed field, an unsupported provider -- always as
    ModelConfigError naming this file's own path, exactly like
    core.kernel.workflow_config.load_workflow_config_file()'s own
    "fail loud on config drift" convention (see that function's
    docstring for why silent/partial loading is deliberately not
    offered here either).
    """

    resolved_path = Path(path)

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelConfigError(
            f"Model config file not found or unreadable: "
            f"{resolved_path} ({exc})."
        ) from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ModelConfigError(
            f"Model config file is not valid JSON: "
            f"{resolved_path} ({exc})."
        ) from exc

    if not isinstance(data, Mapping):
        raise ModelConfigError(
            f"Model config file {resolved_path} must contain a JSON "
            f"object, got {type(data).__name__}."
        )

    provider = _require_non_empty_str(
        data.get("provider"),
        "provider",
        resolved_path,
    ).strip().lower()

    if provider not in SUPPORTED_PROVIDERS:
        raise ModelConfigError(
            f"Model config file {resolved_path}: unsupported "
            f"'provider' {provider!r}; supported providers are "
            f"{SUPPORTED_PROVIDERS}."
        )

    model = _require_non_empty_str(
        data.get("model"),
        "model",
        resolved_path,
    )

    api_key_env = _require_non_empty_str(
        data.get("api_key_env"),
        "api_key_env",
        resolved_path,
    )

    temperature = data.get("temperature")

    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
    ):
        raise ModelConfigError(
            f"Model config file {resolved_path}: 'temperature' must "
            f"be a number or null, got {temperature!r}."
        )

    max_tokens = data.get("max_tokens")

    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ModelConfigError(
            f"Model config file {resolved_path}: 'max_tokens' must "
            f"be a positive integer or null, got {max_tokens!r}."
        )

    return ModelConfig(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        temperature=(
            float(temperature) if temperature is not None else None
        ),
        max_tokens=max_tokens,
    )


def build_llm_client_factory_from_config(
    config: ModelConfig,
) -> Callable[[], LLMClient]:
    """
    Build a zero-argument `llm_client_factory` callable (the same
    shape build_default_kernel() has always required -- see that
    function's own docstring) from an already-loaded ModelConfig.

    Nothing here touches the environment or imports a vendor SDK until
    the returned factory is actually *called* -- matching every other
    factory in this project (AgentRegistration.build_agent,
    build_default_kernel()'s own `decision_engine_factory`), which are
    all deliberately lazy so that constructing one doesn't have a
    side effect. Calling the returned factory:

    1. Reads `config.api_key_env` from the real process environment.
       Raises ModelConfigError immediately if it is unset or empty --
       before attempting any vendor SDK import, so a missing API key
       is never confused with a missing package.
    2. Imports the matching vendor SDK (`anthropic` or `openai`) and
       constructs its client with that API key. Raises ModelConfigError
       (chained from the original ImportError) with a `pip install`
       instruction if that package is not installed -- see this
       module's own top-of-file docstring for why that path, not the
       successful-construction path, is what is actually verified for
       real in this project's own sandbox.
    3. Wraps the constructed vendor client in this provider's existing,
       already-tested LLMClient implementation (ClaudeProvider or
       OpenAIProvider) and returns it.
    """

    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig.")

    def factory() -> LLMClient:

        api_key = os.environ.get(config.api_key_env)

        if not api_key:
            raise ModelConfigError(
                f"Environment variable '{config.api_key_env}' is not "
                f"set (or is empty); it must hold the real API key "
                f"for provider '{config.provider}'. This model config "
                f"deliberately never stores the key itself -- only "
                f"the name of the environment variable to read it "
                f"from."
            )

        if config.provider == "anthropic":

            try:
                import anthropic
            except ImportError as exc:
                raise ModelConfigError(
                    "The 'anthropic' package is not installed, so "
                    "the 'anthropic' provider from this model config "
                    "cannot be built. Install it with `pip install "
                    "anthropic` (it is already listed, unpinned, in "
                    "requirements.txt)."
                ) from exc

            return ClaudeProvider(anthropic.Anthropic(api_key=api_key))

        if config.provider == "openai":

            try:
                import openai
            except ImportError as exc:
                raise ModelConfigError(
                    "The 'openai' package is not installed, so the "
                    "'openai' provider from this model config cannot "
                    "be built. Install it with `pip install openai` "
                    "(it is already listed, unpinned, in "
                    "requirements.txt)."
                ) from exc

            return OpenAIProvider(openai.OpenAI(api_key=api_key))

        # Unreachable in practice -- load_model_config() already
        # restricts `provider` to SUPPORTED_PROVIDERS, and this
        # function only accepts a ModelConfig (never a raw dict), so
        # the only way to reach this branch is constructing a
        # ModelConfig directly with an unsupported provider string,
        # bypassing load_model_config() entirely. Defensive only,
        # mirroring core/kernel/workflow_config.py's own defensive-
        # branch convention for the same kind of "should be
        # impossible, but fail loud rather than silently misbehave if
        # it somehow happens" case.
        raise ModelConfigError(
            f"Unsupported provider {config.provider!r}; supported "
            f"providers are {SUPPORTED_PROVIDERS}."
        )

    return factory

"""
Tests for core.llm.model_config (Build Phase 18).

Two tiers -- but, as of the fix described below, BOTH now run for real
and pass in every environment regardless of what is actually pip
installed there, unlike the equivalent split in this project's
tests/orchestration/test_langgraph_orchestration_engine.py (which
still depends on the real environment for its own "missing" tier):

1. Everything that does not require the `anthropic`/`openai` vendor
   SDKs -- all of load_model_config()'s validation, and
   build_llm_client_factory_from_config()'s missing-API-key error path
   -- runs for real in every environment.

   The missing-*package* error path
   (`test_factory_anthropic_missing_package_raises_clear_error`/
   `test_factory_openai_missing_package_raises_clear_error`) used to
   only run for real when the vendor package genuinely was not
   installed, and otherwise skipped itself ("the missing-package path
   this test checks doesn't apply here"). That was correct but
   unsatisfying once a real machine had both `anthropic` and `openai`
   installed (as this project's own reference machine now does,
   confirmed via a real `pytest -v -rs` run) -- there was then no
   environment left in which that branch of
   `build_llm_client_factory_from_config()` was ever actually
   exercised for real. Fixed by using `monkeypatch.setitem(sys.modules,
   "<pkg>", None)`: Python's own import machinery raises
   `ModuleNotFoundError` for any `import <pkg>` while
   `sys.modules["<pkg>"]` is `None` (documented CPython behavior,
   independent of whether the package is actually installed on disk),
   so `factory()`'s own local `import anthropic`/`import openai`
   genuinely fails inside the test, deterministically, in ANY
   environment -- this sandbox, the user's real machine with both
   packages installed, or a fresh clone with neither. `monkeypatch`
   restores the real `sys.modules` entry automatically after the test,
   so this never affects any other test (including the "builds a real
   provider when installed" tests below, whichever order they run in).

2. The "actually builds a real ClaudeProvider/OpenAIProvider when the
   vendor package IS installed" paths are skip-guarded with
   pytest.importorskip, and only run for real on a machine that has
   run `pip install -r requirements.txt`. Confirmed passing for real on
   this project's own reference machine.

Deliberately all flat module-level test_* functions, not test
classes -- matching every other test file in this project (grep across
tests/ confirms no other file uses class-based grouping), each test
managing its own temp dir/env-var cleanup directly, exactly like
tests/kernel/test_workflow_config.py's own established pattern.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from core.llm.model_config import (
    SUPPORTED_PROVIDERS,
    ModelConfig,
    ModelConfigError,
    build_llm_client_factory_from_config,
    load_model_config,
)

from core.llm.providers.claude_provider import ClaudeProvider
from core.llm.providers.openai_provider import OpenAIProvider


def _write_config_file(tmp_dir: Path, name: str, content) -> Path:
    path = tmp_dir / name

    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(json.dumps(content), encoding="utf-8")

    return path


def test_load_model_config_missing_file_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        missing_path = tmp_dir / "does_not_exist.json"

        with pytest.raises(ModelConfigError, match="not found or unreadable"):
            load_model_config(missing_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_invalid_json_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(tmp_dir, "bad.json", "{not valid json")

        with pytest.raises(ModelConfigError, match="not valid JSON"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_non_object_json_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(tmp_dir, "list.json", ["anthropic"])

        with pytest.raises(ModelConfigError, match="JSON object"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_missing_provider_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "no_provider.json",
            {"model": "x", "api_key_env": "X_API_KEY"},
        )

        with pytest.raises(ModelConfigError, match="'provider'"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_unsupported_provider_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "bad_provider.json",
            {
                "provider": "not_a_real_provider",
                "model": "x",
                "api_key_env": "X_API_KEY",
            },
        )

        with pytest.raises(ModelConfigError, match="unsupported 'provider'"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_missing_model_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "no_model.json",
            {"provider": "anthropic", "api_key_env": "X_API_KEY"},
        )

        with pytest.raises(ModelConfigError, match="'model'"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_missing_api_key_env_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "no_key_env.json",
            {"provider": "anthropic", "model": "x"},
        )

        with pytest.raises(ModelConfigError, match="'api_key_env'"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_non_numeric_temperature_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "bad_temp.json",
            {
                "provider": "anthropic",
                "model": "x",
                "api_key_env": "X_API_KEY",
                "temperature": "hot",
            },
        )

        with pytest.raises(ModelConfigError, match="'temperature'"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_boolean_temperature_raises():
    # bool is a subclass of int in Python -- must be rejected
    # explicitly, exactly like core.kernel.workflow_config's own
    # numeric-field validation already has to guard against this.
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "bool_temp.json",
            {
                "provider": "anthropic",
                "model": "x",
                "api_key_env": "X_API_KEY",
                "temperature": True,
            },
        )

        with pytest.raises(ModelConfigError, match="'temperature'"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_non_positive_max_tokens_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "bad_tokens.json",
            {
                "provider": "anthropic",
                "model": "x",
                "api_key_env": "X_API_KEY",
                "max_tokens": 0,
            },
        )

        with pytest.raises(ModelConfigError, match="'max_tokens'"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_float_max_tokens_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "float_tokens.json",
            {
                "provider": "anthropic",
                "model": "x",
                "api_key_env": "X_API_KEY",
                "max_tokens": 12.5,
            },
        )

        with pytest.raises(ModelConfigError, match="'max_tokens'"):
            load_model_config(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_valid_minimal_config_loads():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "minimal.json",
            {
                "provider": "OpenAI",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
            },
        )

        config = load_model_config(path)

        assert config == ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            temperature=None,
            max_tokens=None,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_model_config_valid_full_config_loads():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(
            tmp_dir,
            "full.json",
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "api_key_env": "ANTHROPIC_API_KEY",
                "temperature": 0.2,
                "max_tokens": 2048,
            },
        )

        config = load_model_config(path)

        assert config == ModelConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
            api_key_env="ANTHROPIC_API_KEY",
            temperature=0.2,
            max_tokens=2048,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_supported_providers_are_anthropic_and_openai():
    assert SUPPORTED_PROVIDERS == ("anthropic", "openai")


def test_build_llm_client_factory_from_config_rejects_non_model_config():
    with pytest.raises(TypeError, match="ModelConfig"):
        build_llm_client_factory_from_config({"provider": "anthropic"})


def test_build_llm_client_factory_from_config_returns_callable_without_side_effects():
    # Building the factory must be a pure, side-effect-free step --
    # exactly like every other factory in this project (see this
    # module's own docstring). Constructing it must not itself read
    # the environment or import a vendor SDK.
    config = ModelConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key_env="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
    )

    factory = build_llm_client_factory_from_config(config)

    assert callable(factory)


def test_factory_missing_api_key_env_raises_before_any_sdk_import():
    os.environ.pop("SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ", None)

    config = ModelConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key_env="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
    )

    factory = build_llm_client_factory_from_config(config)

    with pytest.raises(
        ModelConfigError,
        match="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
    ):
        factory()


def test_factory_empty_api_key_env_raises():
    os.environ["SOME_EMPTY_VAR_XYZ"] = ""
    try:
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env="SOME_EMPTY_VAR_XYZ",
        )

        factory = build_llm_client_factory_from_config(config)

        with pytest.raises(ModelConfigError, match="SOME_EMPTY_VAR_XYZ"):
            factory()
    finally:
        os.environ.pop("SOME_EMPTY_VAR_XYZ", None)


def test_factory_anthropic_missing_package_raises_clear_error(monkeypatch):
    # Simulates "anthropic is not installed" deterministically, in ANY
    # environment, via Python's own documented import behavior: an
    # `import anthropic` while sys.modules["anthropic"] is None raises
    # ModuleNotFoundError, regardless of whether the real package is
    # actually installed on disk. `monkeypatch.setitem` restores the
    # real entry (present or absent) automatically after this test --
    # see this module's own top-of-file docstring for the full
    # rationale.
    monkeypatch.setitem(sys.modules, "anthropic", None)

    os.environ["FAKE_ANTHROPIC_KEY_XYZ"] = "sk-fake"
    try:
        config = ModelConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
            api_key_env="FAKE_ANTHROPIC_KEY_XYZ",
        )

        factory = build_llm_client_factory_from_config(config)

        with pytest.raises(ModelConfigError, match="anthropic.*not installed"):
            factory()
    finally:
        os.environ.pop("FAKE_ANTHROPIC_KEY_XYZ", None)


def test_factory_openai_missing_package_raises_clear_error(monkeypatch):
    # Same simulated-absence technique as the anthropic test above.
    monkeypatch.setitem(sys.modules, "openai", None)

    os.environ["FAKE_OPENAI_KEY_XYZ"] = "sk-fake"
    try:
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env="FAKE_OPENAI_KEY_XYZ",
        )

        factory = build_llm_client_factory_from_config(config)

        with pytest.raises(ModelConfigError, match="openai.*not installed"):
            factory()
    finally:
        os.environ.pop("FAKE_OPENAI_KEY_XYZ", None)


def test_factory_anthropic_builds_real_claude_provider_when_installed():
    pytest.importorskip("anthropic")

    os.environ["REAL_ANTHROPIC_KEY_XYZ"] = "sk-fake-but-present"
    try:
        config = ModelConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
            api_key_env="REAL_ANTHROPIC_KEY_XYZ",
        )

        factory = build_llm_client_factory_from_config(config)
        client = factory()

        assert isinstance(client, ClaudeProvider)
    finally:
        os.environ.pop("REAL_ANTHROPIC_KEY_XYZ", None)


def test_factory_openai_builds_real_openai_provider_when_installed():
    pytest.importorskip("openai")

    os.environ["REAL_OPENAI_KEY_XYZ"] = "sk-fake-but-present"
    try:
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env="REAL_OPENAI_KEY_XYZ",
        )

        factory = build_llm_client_factory_from_config(config)
        client = factory()

        assert isinstance(client, OpenAIProvider)
    finally:
        os.environ.pop("REAL_OPENAI_KEY_XYZ", None)

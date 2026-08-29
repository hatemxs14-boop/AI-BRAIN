"""
Tests for core.llm.model_config (Build Phase 18).

Two tiers, deliberately -- the same split core/orchestration/
langgraph_orchestration_engine.py's own test suite already established
for exactly the same reason (see this project's tests/orchestration/
test_langgraph_orchestration_engine.py):

1. Everything that does not require the `anthropic`/`openai` vendor
   SDKs -- all of load_model_config()'s validation, and
   build_llm_client_factory_from_config()'s missing-API-key and
   missing-package error paths -- runs for real in every environment,
   this sandbox included (neither vendor package is installed here;
   confirmed separately -- there is no package-index access in this
   sandbox at all).

2. The "actually builds a real ClaudeProvider/OpenAIProvider when the
   vendor package IS installed" paths are skip-guarded with
   pytest.importorskip, and only run for real on a machine that has
   run `pip install -r requirements.txt`. Per core/llm/model_config.py's
   own top-of-file docstring, those two branches are written against
   each SDK's documented client-construction API but have never
   executed anywhere yet -- this is the first real verification of
   them, still pending on a real machine, not merely written.

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


def test_factory_anthropic_missing_package_raises_clear_error():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip(
            "anthropic is installed in this environment; the "
            "missing-package path this test checks doesn't apply "
            "here -- see the importorskip-guarded test instead."
        )

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


def test_factory_openai_missing_package_raises_clear_error():
    try:
        import openai  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip(
            "openai is installed in this environment; the "
            "missing-package path this test checks doesn't apply "
            "here -- see the importorskip-guarded test instead."
        )

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

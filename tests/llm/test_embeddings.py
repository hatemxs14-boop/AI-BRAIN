"""
Tests for core.llm.embeddings (Build Phase 28): EmbeddingClient's own
ABC contract, VoyageEmbeddingClient's input validation and response
parsing (against a fake, in-process stand-in for a real
`voyageai.Client`, never the real network), build_embedding_client_
factory()'s two real, actionable error paths (missing API key,
missing vendor package) plus its happy path when `voyageai` genuinely
is installed, and cosine_similarity()'s pure-Python math.

Same two-tier honesty split tests/llm/test_model_config.py's own
top-of-file docstring already documents for
build_llm_client_factory_from_config(): the missing-package path is
verified for real in ANY environment via `monkeypatch.setitem(
sys.modules, "voyageai", None)` (documented CPython import behavior,
independent of whether the package is actually installed on disk),
while the "builds a real VoyageEmbeddingClient when voyageai IS
installed" path is skip-guarded with `pytest.importorskip` and only
runs for real on a machine that has it installed -- this sandbox does
not (confirmed: no PyPI access here), so that one test is expected to
report "skipped" in this sandbox's own `pytest -v -rs` output, exactly
mirroring test_model_config.py's own two anthropic/openai "builds a
real provider" tests.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

from core.llm.embeddings import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    EmbeddingClient,
    EmbeddingConfigError,
    VoyageEmbeddingClient,
    build_embedding_client_factory,
    cosine_similarity,
)


# ---------------------------------------------------------------------
# EmbeddingClient -- ABC contract
# ---------------------------------------------------------------------


def test_embedding_client_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        EmbeddingClient()


# ---------------------------------------------------------------------
# VoyageEmbeddingClient -- construction validation
# ---------------------------------------------------------------------


def test_voyage_embedding_client_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        VoyageEmbeddingClient(object(), model="   ")


def test_voyage_embedding_client_rejects_non_string_model():
    with pytest.raises(ValueError, match="model"):
        VoyageEmbeddingClient(object(), model=123)


# ---------------------------------------------------------------------
# VoyageEmbeddingClient.embed() -- input validation
# ---------------------------------------------------------------------


class _FakeVoyageResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeVoyageVendorClient:
    """
    A minimal, in-process stand-in for a real `voyageai.Client` --
    never touches the network. `.embed()` records the last call it
    received and returns a canned `_FakeVoyageResult`, exactly
    mirroring tests/llm/test_llm_decision_engine.py's own MockLLMClient
    convention (a real object satisfying the vendor client's shape,
    not the real SDK).
    """

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self.last_call = None

    def embed(self, texts, *, model, input_type):
        self.last_call = {
            "texts": texts,
            "model": model,
            "input_type": input_type,
        }
        return _FakeVoyageResult(self._embeddings)


def test_embed_rejects_a_single_string_instead_of_a_sequence():
    client = VoyageEmbeddingClient(
        _FakeVoyageVendorClient([[0.1, 0.2]]), model="voyage-4-lite"
    )

    with pytest.raises(TypeError, match="sequence of strings"):
        client.embed("hello", input_type="document")


def test_embed_rejects_empty_texts():
    client = VoyageEmbeddingClient(
        _FakeVoyageVendorClient([]), model="voyage-4-lite"
    )

    with pytest.raises(ValueError, match="must not be empty"):
        client.embed((), input_type="document")


def test_embed_rejects_a_blank_string_in_texts():
    client = VoyageEmbeddingClient(
        _FakeVoyageVendorClient([[0.1]]), model="voyage-4-lite"
    )

    with pytest.raises(ValueError, match="non-empty string"):
        client.embed(("hello", "   "), input_type="document")


def test_embed_rejects_an_invalid_input_type():
    client = VoyageEmbeddingClient(
        _FakeVoyageVendorClient([[0.1]]), model="voyage-4-lite"
    )

    with pytest.raises(ValueError, match="input_type"):
        client.embed(("hello",), input_type="not-a-real-type")


def test_embed_rejects_a_response_missing_the_embeddings_field():
    vendor = _FakeVoyageVendorClient(None)
    client = VoyageEmbeddingClient(vendor, model="voyage-4-lite")

    with pytest.raises(ValueError, match="no 'embeddings' field"):
        client.embed(("hello",), input_type="document")


def test_embed_rejects_a_vector_count_mismatch():
    vendor = _FakeVoyageVendorClient([[0.1, 0.2]])
    client = VoyageEmbeddingClient(vendor, model="voyage-4-lite")

    with pytest.raises(ValueError, match="expected exactly one vector"):
        client.embed(("hello", "world"), input_type="document")


# ---------------------------------------------------------------------
# VoyageEmbeddingClient.embed() -- happy path (against the fake vendor
# client double -- exercises this module's own parsing/plumbing code
# for real; never the real network, since `voyageai` is not installed
# in this sandbox).
# ---------------------------------------------------------------------


def test_embed_returns_one_tuple_vector_per_input_in_order():
    vendor = _FakeVoyageVendorClient([[0.1, 0.2], [0.3, 0.4]])
    client = VoyageEmbeddingClient(vendor, model="voyage-4-lite")

    result = client.embed(("first", "second"), input_type="document")

    assert result == ((0.1, 0.2), (0.3, 0.4))
    assert vendor.last_call == {
        "texts": ["first", "second"],
        "model": "voyage-4-lite",
        "input_type": "document",
    }


def test_embed_passes_input_type_through_to_the_vendor_client():
    vendor = _FakeVoyageVendorClient([[0.5]])
    client = VoyageEmbeddingClient(vendor, model="voyage-4-lite")

    client.embed(("a query",), input_type="query")

    assert vendor.last_call["input_type"] == "query"


def test_embed_coerces_vector_components_to_float():
    vendor = _FakeVoyageVendorClient([[1, 2]])
    client = VoyageEmbeddingClient(vendor, model="voyage-4-lite")

    result = client.embed(("hello",), input_type="document")

    assert result == ((1.0, 2.0),)
    assert all(isinstance(component, float) for component in result[0])


# ---------------------------------------------------------------------
# build_embedding_client_factory() -- construction validation
# ---------------------------------------------------------------------


def test_supported_embedding_providers_is_voyage_only():
    assert SUPPORTED_EMBEDDING_PROVIDERS == ("voyage",)


def test_factory_rejects_unsupported_provider():
    with pytest.raises(EmbeddingConfigError, match="Unsupported"):
        build_embedding_client_factory(
            provider="not_a_real_provider",
            model="voyage-4-lite",
            api_key_env="SOME_VAR",
        )


def test_factory_rejects_empty_model():
    with pytest.raises(EmbeddingConfigError, match="model"):
        build_embedding_client_factory(
            provider="voyage", model="   ", api_key_env="SOME_VAR"
        )


def test_factory_rejects_empty_api_key_env():
    with pytest.raises(EmbeddingConfigError, match="api_key_env"):
        build_embedding_client_factory(
            provider="voyage", model="voyage-4-lite", api_key_env="   "
        )


def test_factory_returns_callable_without_side_effects():
    # Building the factory must not itself read the environment or
    # import voyageai -- exactly like build_llm_client_factory_from_
    # config()'s own identical precedent.
    factory = build_embedding_client_factory(
        provider="voyage",
        model="voyage-4-lite",
        api_key_env="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
    )

    assert callable(factory)


# ---------------------------------------------------------------------
# build_embedding_client_factory() -- factory() error paths
# ---------------------------------------------------------------------


def test_factory_missing_api_key_env_raises_before_any_sdk_import():
    os.environ.pop("SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ", None)

    factory = build_embedding_client_factory(
        provider="voyage",
        model="voyage-4-lite",
        api_key_env="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
    )

    with pytest.raises(
        EmbeddingConfigError,
        match="SOME_VAR_THAT_IS_DEFINITELY_NOT_SET_XYZ",
    ):
        factory()


def test_factory_empty_api_key_env_raises():
    os.environ["SOME_EMPTY_VOYAGE_VAR_XYZ"] = ""
    try:
        factory = build_embedding_client_factory(
            provider="voyage",
            model="voyage-4-lite",
            api_key_env="SOME_EMPTY_VOYAGE_VAR_XYZ",
        )

        with pytest.raises(
            EmbeddingConfigError, match="SOME_EMPTY_VOYAGE_VAR_XYZ"
        ):
            factory()
    finally:
        os.environ.pop("SOME_EMPTY_VOYAGE_VAR_XYZ", None)


def test_factory_missing_package_raises_clear_error(monkeypatch):
    # Simulates "voyageai is not installed" deterministically, in ANY
    # environment -- see this module's own top-of-file docstring for
    # the full rationale (identical technique to test_model_config.py's
    # own anthropic/openai equivalents).
    monkeypatch.setitem(sys.modules, "voyageai", None)

    os.environ["FAKE_VOYAGE_KEY_XYZ"] = "pa-fake"
    try:
        factory = build_embedding_client_factory(
            provider="voyage",
            model="voyage-4-lite",
            api_key_env="FAKE_VOYAGE_KEY_XYZ",
        )

        with pytest.raises(
            EmbeddingConfigError, match="voyageai.*not installed"
        ):
            factory()
    finally:
        os.environ.pop("FAKE_VOYAGE_KEY_XYZ", None)


def test_factory_builds_real_voyage_embedding_client_when_installed():
    pytest.importorskip("voyageai")

    os.environ["REAL_VOYAGE_KEY_XYZ"] = "pa-fake-but-present"
    try:
        factory = build_embedding_client_factory(
            provider="voyage",
            model="voyage-4-lite",
            api_key_env="REAL_VOYAGE_KEY_XYZ",
        )

        client = factory()

        assert isinstance(client, VoyageEmbeddingClient)
    finally:
        os.environ.pop("REAL_VOYAGE_KEY_XYZ", None)


# ---------------------------------------------------------------------
# cosine_similarity()
# ---------------------------------------------------------------------


def test_cosine_similarity_of_identical_vectors_is_one():
    assert cosine_similarity((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx(1.0)


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_similarity_matches_hand_computed_value():
    a = (1.0, 2.0)
    b = (2.0, 3.0)

    expected = (1.0 * 2.0 + 2.0 * 3.0) / (
        math.sqrt(1.0**2 + 2.0**2) * math.sqrt(2.0**2 + 3.0**2)
    )

    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_similarity_returns_zero_for_a_zero_vector_instead_of_raising():
    assert cosine_similarity((0.0, 0.0), (1.0, 2.0)) == 0.0
    assert cosine_similarity((1.0, 2.0), (0.0, 0.0)) == 0.0
    assert cosine_similarity((0.0, 0.0), (0.0, 0.0)) == 0.0


def test_cosine_similarity_rejects_an_empty_vector():
    with pytest.raises(ValueError, match="non-empty"):
        cosine_similarity((), (1.0,))

    with pytest.raises(ValueError, match="non-empty"):
        cosine_similarity((1.0,), ())


def test_cosine_similarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity((1.0, 2.0), (1.0, 2.0, 3.0))

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Callable, Sequence


# ---------------------------------------------------------------------
# Build Phase 28 -- real semantic (embedding-based) text similarity.
#
# The problem this solves: this project's most-repeated, most honestly
# documented gap is "not real NLU" -- MemoryStore.search()
# (core/memory/memory_store.py, Build Phase 14), Kernel CLASSIFY's
# agent-routing keyword vocabularies (core/kernel/default_kernel.py,
# Build Phase 8), the `topic_drift` guardrail check (core/agents/
# guardrails.py, Build Phase 23), and ModelTierRouter's own complexity
# heuristic (core/llm/model_tier.py, Build Phase 27) all match on
# whole-word/substring keyword overlap, never real semantic similarity.
# This module is the first real, non-fabricated foundation for closing
# that gap: a provider-independent `EmbeddingClient` interface (mirrors
# core/llm/llm_client.py's own LLMClient interface exactly, one layer
# over from generation to embedding) plus a real Voyage AI
# implementation and a dependency-free cosine-similarity helper.
#
# Provider choice: Anthropic itself has no embeddings model or API of
# its own -- Voyage AI is Anthropic's own recommended embeddings
# partner (see https://platform.claude.com/docs/en/build-with-claude/
# embeddings), so this module wraps Voyage AI's real, documented
# Python client (`voyageai.Client().embed(...)`) rather than inventing
# a fabricated in-house scheme or silently defaulting to a different,
# unrelated provider. Chosen explicitly by the user, after being told
# the real, current per-call cost (a small fraction of a cent per
# million tokens for the cheapest tier -- negligible next to a full
# LLM generation call) and the real trade-off against a fully local,
# zero-marginal-cost model: the user's own priority is minimizing
# actual financial/token spend, not minimizing dependency count for
# its own sake, so a cheap, well-supported managed API is the right
# call here, not a violation of this project's "minimize dependencies"
# constraint applied dogmatically.
#
# Like core/llm/model_config.py's own two provider branches, this
# module is honest about what has and hasn't run for real in this
# sandbox: `voyageai` is not installed here (no PyPI access in this
# sandbox), so VoyageEmbeddingClient's real embed() call against a
# genuine `voyageai.Client` has never executed anywhere in this
# sandbox. What IS verified for real here (see tests/llm/
# test_embeddings.py) is every piece of this module that does not
# require the vendor SDK: VoyageEmbeddingClient's own input validation
# and response-parsing logic (exercised against a fake, in-process
# stand-in client, not the real network), cosine_similarity()'s pure-
# Python math, and build_embedding_client_factory()'s returned
# factory's own two real, actionable error paths -- a missing
# environment variable, and the vendor package genuinely not being
# installed (exercised for real here, since it genuinely isn't) --
# exactly mirroring build_llm_client_factory_from_config()'s own
# "verified error paths, unverified happy path" honesty.
#
# Never stores a real API key anywhere in this module -- only the
# *name* of the environment variable to read one from, at the moment a
# real client is actually built (never at import time, never on this
# module's own dataclasses/functions) -- the same standing rule
# core/llm/model_config.py's own ModelConfig already established for
# LLM provider keys, applied here to VOYAGE_API_KEY.
# ---------------------------------------------------------------------


class EmbeddingClient(ABC):
    """
    Provider-independent interface for turning text into embedding
    vectors -- the same role core.llm.llm_client.LLMClient plays for
    text generation, one layer over.

    This layer does not:

    - decide agent actions
    - execute tools
    - authorize operations
    - access the Security Layer
    - contain provider-specific business logic
    """

    @abstractmethod
    def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
    ) -> tuple[tuple[float, ...], ...]:
        """
        Embed each string in `texts`, in order, returning one vector
        (a tuple of floats) per input.

        `input_type` is either "document" (text being indexed/stored,
        e.g. a MemoryRecord's own content) or "query" (text a caller is
        searching with) -- Voyage AI's own documented distinction for
        asymmetric retrieval quality: a corpus and a query embedded
        with the matching `input_type` retrieve measurably better than
        embedding both the same way. Concrete implementations should
        honor this distinction rather than ignoring it.
        """
        raise NotImplementedError


class VoyageEmbeddingClient(EmbeddingClient):
    """
    Voyage AI implementation of EmbeddingClient.

    Constructor takes an already-constructed `voyageai.Client` instance
    -- never touches an API key or the environment itself -- exactly
    mirroring ClaudeProvider/OpenAIProvider's own "wrap an
    already-built vendor client" convention (core/llm/providers/
    claude_provider.py, core/llm/providers/openai_provider.py, Build
    Phase 4). Use `build_embedding_client_factory()` below to construct
    one from a config-style provider/model/api_key_env triple, the same
    two-step shape core/llm/model_config.py's own
    build_llm_client_factory_from_config() already established.
    """

    _VALID_INPUT_TYPES = ("document", "query")

    def __init__(self, client, *, model: str) -> None:

        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "VoyageEmbeddingClient.model must be a non-empty "
                "string."
            )

        self.client = client
        self.model = model

    def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
    ) -> tuple[tuple[float, ...], ...]:

        if isinstance(texts, (str, bytes)) or not isinstance(
            texts, Sequence
        ):
            raise TypeError(
                "texts must be a sequence of strings, not a single "
                "string."
            )

        texts = tuple(texts)

        if not texts:
            raise ValueError("texts must not be empty.")

        if not all(
            isinstance(text, str) and text.strip() for text in texts
        ):
            raise ValueError(
                "Every item in texts must be a non-empty string."
            )

        if input_type not in self._VALID_INPUT_TYPES:
            raise ValueError(
                "input_type must be one of "
                f"{self._VALID_INPUT_TYPES}, got {input_type!r}."
            )

        result = self.client.embed(
            list(texts),
            model=self.model,
            input_type=input_type,
        )

        embeddings = getattr(result, "embeddings", None)

        if embeddings is None:
            raise ValueError(
                "Voyage AI's embed() response carries no 'embeddings' "
                "field -- cannot use this result."
            )

        if len(embeddings) != len(texts):
            raise ValueError(
                "Voyage AI's embed() returned "
                f"{len(embeddings)} vector(s) for {len(texts)} input "
                "text(s) -- expected exactly one vector per input."
            )

        return tuple(
            tuple(float(component) for component in vector)
            for vector in embeddings
        )


class EmbeddingConfigError(ValueError):
    """
    Raised when an embedding provider/model configuration (or the
    environment it references) is missing, malformed, or otherwise
    invalid.

    Deliberately a ValueError subclass, matching core.llm.model_config.
    ModelConfigError's own convention for the identical reason: this
    always signals a human-fixable configuration mistake -- an
    unsupported provider, a missing environment variable, an
    uninstalled package -- never an internal bug in this project's own
    code.
    """


# The only embedding provider this module currently implements. A
# plain tuple, not an enum, matching core.llm.model_config.
# SUPPORTED_PROVIDERS's own "adding a second provider later only means
# one more branch plus one more entry here" convention.
SUPPORTED_EMBEDDING_PROVIDERS = ("voyage",)


def build_embedding_client_factory(
    *,
    provider: str,
    model: str,
    api_key_env: str,
) -> Callable[[], EmbeddingClient]:
    """
    Build a zero-argument factory callable returning a fresh
    EmbeddingClient -- the exact same lazy-factory shape every other
    factory in this project already uses (AgentRegistration.build_agent,
    build_default_kernel()'s own decision_engine_factory,
    core.llm.model_config.build_llm_client_factory_from_config()).

    Nothing here touches the environment or imports a vendor SDK until
    the returned factory is actually *called*. Calling it:

    1. Reads `api_key_env` from the real process environment. Raises
       EmbeddingConfigError immediately if it is unset or empty --
       before attempting any vendor SDK import, so a missing API key
       is never confused with a missing package (mirrors
       build_llm_client_factory_from_config()'s own ordering).
    2. Imports the `voyageai` package and constructs its client with
       that API key. Raises EmbeddingConfigError (chained from the
       original ImportError) with a `pip install` instruction if that
       package is not installed -- see this module's own top-of-file
       docstring for why that path, not the successful-construction
       path, is what is actually verified for real in this project's
       own sandbox.
    3. Wraps the constructed vendor client in VoyageEmbeddingClient and
       returns it.
    """

    if (
        not isinstance(provider, str)
        or provider.strip().lower() not in SUPPORTED_EMBEDDING_PROVIDERS
    ):
        raise EmbeddingConfigError(
            f"Unsupported embedding provider {provider!r}; supported "
            f"providers are {SUPPORTED_EMBEDDING_PROVIDERS}."
        )

    if not isinstance(model, str) or not model.strip():
        raise EmbeddingConfigError(
            "model must be a non-empty string."
        )

    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise EmbeddingConfigError(
            "api_key_env must be a non-empty string."
        )

    def factory() -> EmbeddingClient:

        api_key = os.environ.get(api_key_env)

        if not api_key:
            raise EmbeddingConfigError(
                f"Environment variable '{api_key_env}' is not set (or "
                "is empty); it must hold the real Voyage AI API key. "
                "This module deliberately never stores the key itself "
                "-- only the name of the environment variable to read "
                "it from."
            )

        try:
            import voyageai
        except ImportError as exc:
            raise EmbeddingConfigError(
                "The 'voyageai' package is not installed, so the "
                "'voyage' embedding provider cannot be built. Install "
                "it with `pip install voyageai`."
            ) from exc

        return VoyageEmbeddingClient(
            voyageai.Client(api_key=api_key),
            model=model,
        )

    return factory


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Dependency-free cosine similarity between two equal-length numeric
    vectors, in [-1.0, 1.0] for any two non-zero vectors.

    Deliberately plain Python (`math.sqrt`, a couple of `sum()`
    comprehensions) rather than numpy -- this project has no numeric-
    array dependency anywhere else, and a single pairwise similarity
    computation over embedding vectors of a few thousand components at
    most has no real need for one. Returns 0.0 (never raises a
    division-by-zero) when either vector is the all-zero vector, since
    "similarity to a vector with no direction" is not meaningfully
    defined as anything else.

    Raises ValueError for empty vectors or a length mismatch -- two
    embeddings from the same model always have the same, fixed
    dimensionality, so a mismatch here always signals a real caller
    bug (e.g. comparing vectors from two different models), never an
    expected runtime condition to silently tolerate.
    """

    if not a or not b:
        raise ValueError(
            "cosine_similarity() requires two non-empty vectors."
        )

    if len(a) != len(b):
        raise ValueError(
            "cosine_similarity() requires two vectors of the same "
            f"length; got {len(a)} and {len(b)}."
        )

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)

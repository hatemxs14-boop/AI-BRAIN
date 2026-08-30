"""
Tests for core.llm.model_tier (Build Phase 27): ModelTierRouter's own
construction validation, and route()'s complexity-heuristic behavior --
whole-word/phrase keyword matching (never a plain substring check, the
same convention core.kernel.default_kernel's own agent-routing keyword
vocabularies established) plus a word-count fallback.
"""
from __future__ import annotations

import pytest

from core.llm.model_tier import (
    DEFAULT_COMPLEXITY_KEYWORDS,
    DEFAULT_SIMPLE_MAX_WORDS,
    ModelTierRouter,
)


# ---------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------


def test_model_tier_router_valid_construction():
    router = ModelTierRouter(
        simple_model="cheap-model",
        complex_model="expensive-model",
    )

    assert router.simple_model == "cheap-model"
    assert router.complex_model == "expensive-model"
    assert router.complexity_keywords == DEFAULT_COMPLEXITY_KEYWORDS
    assert router.simple_max_words == DEFAULT_SIMPLE_MAX_WORDS


def test_model_tier_router_rejects_empty_simple_model():
    with pytest.raises(ValueError, match="simple_model"):
        ModelTierRouter(simple_model="   ", complex_model="expensive-model")


def test_model_tier_router_rejects_non_string_simple_model():
    with pytest.raises(ValueError, match="simple_model"):
        ModelTierRouter(simple_model=123, complex_model="expensive-model")


def test_model_tier_router_rejects_empty_complex_model():
    with pytest.raises(ValueError, match="complex_model"):
        ModelTierRouter(simple_model="cheap-model", complex_model="")


def test_model_tier_router_rejects_non_tuple_complexity_keywords():
    with pytest.raises(TypeError, match="complexity_keywords"):
        ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
            complexity_keywords=["analyze", "compare"],
        )


def test_model_tier_router_rejects_empty_string_in_complexity_keywords():
    with pytest.raises(TypeError, match="complexity_keywords"):
        ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
            complexity_keywords=("analyze", "   "),
        )


def test_model_tier_router_rejects_non_int_simple_max_words():
    with pytest.raises(TypeError, match="simple_max_words"):
        ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
            simple_max_words="12",
        )


def test_model_tier_router_rejects_bool_simple_max_words():
    # bool is a subclass of int -- must be rejected explicitly, same
    # convention TokenBudget.max_total_tokens already established.
    with pytest.raises(TypeError, match="simple_max_words"):
        ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
            simple_max_words=True,
        )


def test_model_tier_router_rejects_zero_simple_max_words():
    with pytest.raises(ValueError, match="simple_max_words"):
        ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
            simple_max_words=0,
        )


def test_model_tier_router_rejects_negative_simple_max_words():
    with pytest.raises(ValueError, match="simple_max_words"):
        ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
            simple_max_words=-1,
        )


# ---------------------------------------------------------------------
# route() -- input validation
# ---------------------------------------------------------------------


def test_route_rejects_non_string_task_text():
    router = ModelTierRouter(
        simple_model="cheap-model", complex_model="expensive-model"
    )

    with pytest.raises(TypeError, match="task_text"):
        router.route(12345)


def test_route_rejects_empty_task_text():
    router = ModelTierRouter(
        simple_model="cheap-model", complex_model="expensive-model"
    )

    with pytest.raises(ValueError, match="task_text"):
        router.route("   ")


# ---------------------------------------------------------------------
# route() -- the actual heuristic
# ---------------------------------------------------------------------


def test_route_sends_a_short_plain_task_to_the_simple_tier():
    router = ModelTierRouter(
        simple_model="cheap-model", complex_model="expensive-model"
    )

    decision = router.route("Search AI agents.")

    assert decision.tier == "simple"
    assert decision.model == "cheap-model"
    assert "3 word(s)" in decision.reason


def test_route_sends_a_task_with_a_complexity_keyword_to_the_complex_tier():
    router = ModelTierRouter(
        simple_model="cheap-model", complex_model="expensive-model"
    )

    decision = router.route("Please give a comprehensive summary.")

    assert decision.tier == "complex"
    assert decision.model == "expensive-model"
    assert "keyword" in decision.reason


def test_route_matches_complexity_keywords_case_insensitively():
    router = ModelTierRouter(
        simple_model="cheap-model", complex_model="expensive-model"
    )

    decision = router.route("Please COMPARE these two options.")

    assert decision.tier == "complex"
    assert decision.model == "expensive-model"


def test_route_does_not_match_a_keyword_embedded_in_a_longer_word():
    # "analyze" must not match inside "reanalyze" -- the same
    # \bphrase\b whole-word convention core/kernel/default_kernel.py's
    # own docstring documents fixing for "find" matching inside
    # "finding"/"findings" (Build Phase 8's own routing bug). A short,
    # otherwise-plain task containing only the embedded form must still
    # land in the simple tier.
    router = ModelTierRouter(
        simple_model="cheap-model", complex_model="expensive-model"
    )

    decision = router.route("Please reanalyze this.")

    assert decision.tier == "simple"
    assert decision.model == "cheap-model"


def test_route_sends_a_long_plain_task_to_the_complex_tier_by_word_count():
    router = ModelTierRouter(
        simple_model="cheap-model",
        complex_model="expensive-model",
        simple_max_words=5,
    )

    decision = router.route("This task has exactly seven plain words.")

    assert decision.tier == "complex"
    assert decision.model == "expensive-model"
    assert "exceeding simple_max_words=5" in decision.reason


def test_route_treats_a_task_at_exactly_simple_max_words_as_simple():
    router = ModelTierRouter(
        simple_model="cheap-model",
        complex_model="expensive-model",
        simple_max_words=5,
    )

    decision = router.route("One two three four five")

    assert decision.tier == "simple"
    assert decision.model == "cheap-model"


def test_route_treats_a_task_one_word_over_the_limit_as_complex():
    router = ModelTierRouter(
        simple_model="cheap-model",
        complex_model="expensive-model",
        simple_max_words=5,
    )

    decision = router.route("One two three four five six")

    assert decision.tier == "complex"
    assert decision.model == "expensive-model"


def test_route_accepts_a_custom_complexity_keywords_tuple():
    router = ModelTierRouter(
        simple_model="cheap-model",
        complex_model="expensive-model",
        complexity_keywords=("urgent",),
    )

    # "comprehensive" is no longer a signal for this router -- only its
    # own custom vocabulary is.
    plain = router.route("Please give a comprehensive summary.")
    assert plain.tier == "simple"

    urgent = router.route("This is urgent.")
    assert urgent.tier == "complex"
    assert urgent.model == "expensive-model"

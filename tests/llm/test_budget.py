"""
Tests for core.llm.budget (Build Phase 26): TokenBudget's own
construction validation, and exceeded_by()'s "None means unknown,
never a fabricated violation" behavior -- the same precedent
combine_token_usage() already established for summing, applied here
to a hard spend ceiling instead.
"""
from __future__ import annotations

import pytest

from core.llm.budget import TokenBudget
from core.llm.token_usage import TokenUsage


def test_token_budget_valid_construction():
    budget = TokenBudget(max_total_tokens=100)

    assert budget.max_total_tokens == 100


def test_token_budget_rejects_non_int_max_total_tokens():
    with pytest.raises(TypeError, match="max_total_tokens"):
        TokenBudget(max_total_tokens="100")


def test_token_budget_rejects_bool_max_total_tokens():
    # bool is a subclass of int in Python -- must be rejected
    # explicitly, matching every other numeric-field validation
    # already established in this project (e.g. TokenUsage's own
    # field checks).
    with pytest.raises(TypeError, match="max_total_tokens"):
        TokenBudget(max_total_tokens=True)


def test_token_budget_rejects_zero_max_total_tokens():
    with pytest.raises(ValueError, match="max_total_tokens"):
        TokenBudget(max_total_tokens=0)


def test_token_budget_rejects_negative_max_total_tokens():
    with pytest.raises(ValueError, match="max_total_tokens"):
        TokenBudget(max_total_tokens=-5)


def test_exceeded_by_returns_false_when_usage_is_none():
    budget = TokenBudget(max_total_tokens=100)

    assert budget.exceeded_by(None) is False


def test_exceeded_by_returns_false_when_usage_is_under_the_cap():
    budget = TokenBudget(max_total_tokens=100)

    usage = TokenUsage(
        prompt_tokens=40,
        completion_tokens=20,
        total_tokens=60,
    )

    assert budget.exceeded_by(usage) is False


def test_exceeded_by_returns_true_when_usage_exactly_reaches_the_cap():
    budget = TokenBudget(max_total_tokens=100)

    usage = TokenUsage(
        prompt_tokens=60,
        completion_tokens=40,
        total_tokens=100,
    )

    assert budget.exceeded_by(usage) is True


def test_exceeded_by_returns_true_when_usage_passes_the_cap():
    budget = TokenBudget(max_total_tokens=100)

    usage = TokenUsage(
        prompt_tokens=80,
        completion_tokens=50,
        total_tokens=130,
    )

    assert budget.exceeded_by(usage) is True


def test_exceeded_by_rejects_non_token_usage_values():
    budget = TokenBudget(max_total_tokens=100)

    with pytest.raises(TypeError, match="TokenUsage or None"):
        budget.exceeded_by("not a usage")

"""
Tests for core.llm.token_usage (Build Phase 19): TokenUsage's own
validation, and combine_token_usage()'s "None means unknown, never
zero" summing behavior.
"""
from __future__ import annotations

import pytest

from core.llm.token_usage import TokenUsage, combine_token_usage


def test_token_usage_valid_construction():
    usage = TokenUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_token_usage_rejects_non_int_field():
    with pytest.raises(TypeError, match="prompt_tokens"):
        TokenUsage(
            prompt_tokens="10",
            completion_tokens=5,
            total_tokens=15,
        )


def test_token_usage_rejects_bool_field():
    # bool is a subclass of int in Python -- must be rejected
    # explicitly, matching every other numeric-field validation
    # already established in this project (e.g. core.llm.model_config.
    # load_model_config's own 'temperature'/'max_tokens' checks).
    with pytest.raises(TypeError, match="completion_tokens"):
        TokenUsage(
            prompt_tokens=10,
            completion_tokens=True,
            total_tokens=15,
        )


def test_token_usage_rejects_negative_field():
    with pytest.raises(ValueError, match="total_tokens"):
        TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=-1,
        )


def test_token_usage_allows_zero():
    usage = TokenUsage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    assert usage.total_tokens == 0


def test_combine_token_usage_with_no_args_returns_none():
    assert combine_token_usage() is None


def test_combine_token_usage_with_all_none_returns_none():
    assert combine_token_usage(None, None, None) is None


def test_combine_token_usage_single_usage_returns_equivalent_usage():
    usage = TokenUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    combined = combine_token_usage(usage)

    assert combined == usage


def test_combine_token_usage_sums_multiple_real_usages():
    first = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    second = TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28)

    combined = combine_token_usage(first, second)

    assert combined == TokenUsage(
        prompt_tokens=30,
        completion_tokens=13,
        total_tokens=43,
    )


def test_combine_token_usage_treats_none_as_unknown_not_zero():
    # A None among real usages must not be treated as "0 tokens" --
    # the real, partial sum from the other usage(s) must still come
    # through untouched.
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    combined = combine_token_usage(None, usage, None)

    assert combined == usage


def test_combine_token_usage_rejects_non_token_usage_values():
    with pytest.raises(TypeError, match="TokenUsage or None"):
        combine_token_usage("not a usage")

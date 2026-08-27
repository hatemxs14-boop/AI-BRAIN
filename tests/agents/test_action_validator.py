from core.agents.action_validator import (
    AgentActionValidator,
)

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)


def test_valid_invoke_tool_action():

    validator = AgentActionValidator()

    action = AgentAction(
        action_type=AgentActionType.INVOKE_TOOL,
        tool_id="web_search",
        inputs={
            "query": "AI agents",
        },
        reason="Search for information.",
    )

    result = validator.validate(action)

    assert result.valid is True
    assert result.errors == ()


def test_validator_accepts_valid_complete_action():

    validator = AgentActionValidator()

    action = AgentAction(
        action_type=AgentActionType.COMPLETE,
        reason="Task completed.",
    )

    result = validator.validate(action)

    assert result.valid is True
    assert result.errors == ()


def test_validator_accepts_valid_fail_action():

    validator = AgentActionValidator()

    action = AgentAction(
        action_type=AgentActionType.FAIL,
        reason="Task failed.",
    )

    result = validator.validate(action)

    assert result.valid is True
    assert result.errors == ()


def test_invoke_tool_with_empty_tool_id_is_rejected_by_action_contract():

    try:
        AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="",
            inputs={
                "query": "AI agents",
            },
            reason="Search.",
        )

    except ValueError as exc:

        assert (
            "tool_id must not be empty"
            in str(exc)
        )

    else:

        raise AssertionError(
            "Invalid AgentAction was accepted."
        )


def test_invoke_tool_with_invalid_inputs_is_rejected_by_action_contract():

    try:
        AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="web_search",
            inputs="invalid",
            reason="Search.",
        )

    except TypeError as exc:

        assert (
            "inputs must be a dictionary"
            in str(exc)
        )

    else:

        raise AssertionError(
            "Invalid AgentAction was accepted."
        )


def test_non_tool_action_with_tool_id_is_rejected_by_action_contract():

    try:
        AgentAction(
            action_type=AgentActionType.COMPLETE,
            tool_id="web_search",
            reason="Complete.",
        )

    except ValueError as exc:

        assert (
            "tool_id must be None"
            in str(exc)
        )

    else:

        raise AssertionError(
            "Invalid AgentAction was accepted."
        )


def test_non_tool_action_with_inputs_is_rejected_by_action_contract():

    try:
        AgentAction(
            action_type=AgentActionType.FAIL,
            inputs={
                "unexpected": True,
            },
            reason="Fail.",
        )

    except ValueError as exc:

        assert (
            "inputs must be None"
            in str(exc)
        )

    else:

        raise AssertionError(
            "Invalid AgentAction was accepted."
        )


def test_non_action_is_rejected():

    validator = AgentActionValidator()

    result = validator.validate(
        "not an action"
    )

    assert result.valid is False

    assert (
        "AgentAction"
        in result.errors[0]
    )
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)


@dataclass(frozen=True)
class ActionValidationResult:
    """
    Result of AgentAction validation.
    """

    valid: bool
    errors: tuple[str, ...] = ()


class AgentActionValidator:
    """
    Validates AgentAction before it reaches AgentCore.

    This layer validates the action structure.

    It does not:

    - execute tools
    - authorize actions
    - access executors
    - make security decisions
    - replace the Tool Gateway
    """

    def validate(
        self,
        action: Any,
    ) -> ActionValidationResult:

        if not isinstance(
            action,
            AgentAction,
        ):
            return ActionValidationResult(
                valid=False,
                errors=(
                    "Action must be an AgentAction.",
                ),
            )

        errors: list[str] = []

        if not isinstance(
            action.action_type,
            AgentActionType,
        ):
            errors.append(
                "Invalid action type."
            )

        if (
            action.action_type
            == AgentActionType.INVOKE_TOOL
        ):
            self._validate_tool_action(
                action,
                errors,
            )

        elif (
            action.action_type
            == AgentActionType.COMPLETE
        ):
            self._validate_terminal_action(
                action,
                errors,
            )

        elif (
            action.action_type
            == AgentActionType.FAIL
        ):
            self._validate_terminal_action(
                action,
                errors,
            )

        else:
            errors.append(
                "Unsupported action type."
            )

        return ActionValidationResult(
            valid=not errors,
            errors=tuple(errors),
        )

    @staticmethod
    def _validate_tool_action(
        action: AgentAction,
        errors: list[str],
    ) -> None:

        if not isinstance(
            action.tool_id,
            str,
        ) or not action.tool_id.strip():
            errors.append(
                "INVOKE_TOOL requires a valid tool_id."
            )

        if not isinstance(
            action.inputs,
            dict,
        ):
            errors.append(
                "INVOKE_TOOL inputs must be a dictionary."
            )

    @staticmethod
    def _validate_terminal_action(
        action: AgentAction,
        errors: list[str],
    ) -> None:

        if action.tool_id is not None:
            errors.append(
                "Terminal actions must not specify tool_id."
            )

        if action.inputs is not None:
            errors.append(
                "Terminal actions must not specify inputs."
            )
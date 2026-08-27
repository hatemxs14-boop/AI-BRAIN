from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentActionType(str, Enum):
    INVOKE_TOOL = "INVOKE_TOOL"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


@dataclass(frozen=True)
class AgentAction:
    action_type: AgentActionType
    tool_id: str | None = None
    inputs: dict[str, Any] | None = None
    reason: str | None = None
    approved: bool | None = None
    approved_by: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.action_type,
            AgentActionType,
        ):
            raise TypeError(
                "action_type must be an AgentActionType."
            )

        if self.action_type == AgentActionType.INVOKE_TOOL:
            if not isinstance(
                self.tool_id,
                str,
            ):
                raise TypeError(
                    "tool_id must be a string for INVOKE_TOOL."
                )

            if not self.tool_id.strip():
                raise ValueError(
                    "tool_id must not be empty for INVOKE_TOOL."
                )

            if not isinstance(
                self.inputs,
                dict,
            ):
                raise TypeError(
                    "inputs must be a dictionary for INVOKE_TOOL."
                )

        else:
            if self.tool_id is not None:
                raise ValueError(
                    "tool_id must be None for non-tool actions."
                )

            if self.inputs is not None:
                raise ValueError(
                    "inputs must be None for non-tool actions."
                )

        if self.reason is not None:
            if not isinstance(
                self.reason,
                str,
            ):
                raise TypeError(
                    "reason must be a string or None."
                )

            if not self.reason.strip():
                raise ValueError(
                    "reason must not be empty."
                )

        if self.approved is not None:
            if not isinstance(
                self.approved,
                bool,
            ):
                raise TypeError(
                    "approved must be a boolean or None."
                )

        if self.approved_by is not None:
            if not isinstance(
                self.approved_by,
                str,
            ):
                raise TypeError(
                    "approved_by must be a string or None."
                )

            if not self.approved_by.strip():
                raise ValueError(
                    "approved_by must not be empty."
                )

        if self.metadata is not None:
            if not isinstance(
                self.metadata,
                dict,
            ):
                raise TypeError(
                    "metadata must be a dictionary or None."
                )
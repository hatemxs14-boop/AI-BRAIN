from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolInvocation:
    """
    Standard invocation contract used by Agents when requesting
    execution of a tool.

    This object contains the request data only.

    It does not execute tools.
    It does not grant permissions.
    It does not perform authorization.
    """

    subject: str
    tool_id: str
    inputs: dict[str, Any]

    approved: bool | None = None
    approved_by: str | None = None

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        Validate the structural contract of the invocation.
        """

        if not isinstance(self.subject, str):
            raise TypeError(
                "subject must be a string."
            )

        if not self.subject.strip():
            raise ValueError(
                "subject must not be empty."
            )

        if not isinstance(self.tool_id, str):
            raise TypeError(
                "tool_id must be a string."
            )

        if not self.tool_id.strip():
            raise ValueError(
                "tool_id must not be empty."
            )

        if not isinstance(self.inputs, dict):
            raise TypeError(
                "inputs must be a dictionary."
            )

        if self.approved is not None and not isinstance(
            self.approved,
            bool,
        ):
            raise TypeError(
                "approved must be a boolean or None."
            )

        if self.approved_by is not None:
            if not isinstance(self.approved_by, str):
                raise TypeError(
                    "approved_by must be a string or None."
                )

            if not self.approved_by.strip():
                raise ValueError(
                    "approved_by must not be empty."
                )

        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise TypeError(
                    "metadata must be a dictionary or None."
                )
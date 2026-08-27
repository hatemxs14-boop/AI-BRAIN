from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """
    Runtime context available to the Agent decision process.

    Stores information needed across execution steps.

    Responsibilities:

    - store the current task
    - store available tools
    - count execution steps
    - store tool results
    - store agent metadata

    Does not:

    - execute tools
    - access executors
    - make security decisions
    - authorize actions
    """

    task: str

    step_count: int = 0

    available_tools: list[dict[str, Any]] = field(
        default_factory=list
    )

    tool_results: list[Any] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def record_step(
        self,
    ) -> None:
        """
        Record one completed execution-loop step.
        """

        self.step_count += 1

    def record_tool_result(
        self,
        result: Any,
    ) -> None:
        """
        Record a ToolExecutionResult or compatible result.
        """

        self.tool_results.append(
            result
        )

    def set_available_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> None:
        """
        Replace the currently available tool descriptions.

        Tool discovery is informational only.
        This method does not authorize tool execution.
        """

        if not isinstance(
            tools,
            list,
        ):
            raise TypeError(
                "tools must be a list."
            )

        validated_tools: list[dict[str, Any]] = []

        for tool in tools:

            if not isinstance(
                tool,
                dict,
            ):
                raise TypeError(
                    "Each available tool must be a dictionary."
                )

            tool_id = tool.get(
                "id"
            )

            if not isinstance(
                tool_id,
                str,
            ) or not tool_id.strip():

                raise ValueError(
                    "Each available tool must contain "
                    "a non-empty string 'id'."
                )

            validated_tools.append(
                dict(tool)
            )

        self.available_tools = validated_tools

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Store runtime metadata.
        """

        if not isinstance(
            key,
            str,
        ) or not key:

            raise ValueError(
                "metadata key must be a non-empty string."
            )

        self.metadata[key] = value

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve runtime metadata.
        """

        return self.metadata.get(
            key,
            default,
        )
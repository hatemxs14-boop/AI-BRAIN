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

    def record_step(self) -> None:
        self.step_count += 1

    def record_tool_result(
        self,
        result: Any,
    ) -> None:
        self.tool_results.append(result)

    def set_available_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> None:
        self.available_tools = list(tools)

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.metadata[key] = value

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.metadata.get(
            key,
            default,
        )
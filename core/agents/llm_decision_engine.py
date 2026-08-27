from __future__ import annotations

import json
from typing import Any

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_context import (
    AgentContext,
)

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.llm.llm_client import (
    LLMClient,
)

from core.llm.llm_request import (
    LLMMessage,
    LLMRequest,
)

from core.tools.engine.tool_gateway import (
    ToolExecutionResult,
)


class LLMDecisionEngine(AgentDecisionEngine):
    """
    LLM-backed decision engine for AI-BRAIN.

    Responsibilities:

    - provide the current AgentContext to the LLM
    - expose only discovered tools
    - ask the LLM for exactly one AgentAction
    - parse the LLM response
    - perform structural validation
    - return an AgentAction to the AgentExecutionLoop

    This class NEVER:

    - executes tools
    - accesses executors
    - performs authorization
    - makes security decisions
    - grants permissions
    - bypasses AgentCore
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:

        if not isinstance(
            client,
            LLMClient,
        ):
            raise TypeError(
                "client must implement LLMClient."
            )

        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def decide(
        self,
        context: AgentContext,
    ) -> AgentAction:
        """
        Ask the LLM for the next AgentAction.
        """

        if not isinstance(
            context,
            AgentContext,
        ):
            raise TypeError(
                "context must be an AgentContext."
            )

        request = self._build_request(
            context
        )

        response = self.client.generate(
            request
        )

        if response is None:
            raise ValueError(
                "LLM returned no response."
            )

        content = getattr(
            response,
            "content",
            None,
        )

        if not isinstance(
            content,
            str,
        ):
            raise TypeError(
                "LLM response content must be a string."
            )

        return self._parse_action(
            content
        )

    def _build_request(
        self,
        context: AgentContext,
    ) -> LLMRequest:
        """
        Build a provider-independent LLM request.

        The LLM receives:

        - current task
        - execution step
        - available tools
        - previous tool results

        The LLM only proposes an action.

        It does not receive executor access.
        """

        available_tools = context.get_metadata(
            "available_tools",
            [],
        )

        if not isinstance(
            available_tools,
            list,
        ):
            available_tools = []

        tool_results = [
            self._serialize_tool_result(
                result
            )
            for result in context.tool_results
        ]

        context_data: dict[str, Any] = {
            "task": context.task,
            "step_count": context.step_count,
            "available_tools": available_tools,
            "tool_results": tool_results,
        }

        system_message = LLMMessage(
            role="system",
            content=(
                "You are the decision engine of AI-BRAIN. "
                "Your only responsibility is to choose the next "
                "agent action. "
                "\n\n"
                "You do NOT execute tools yourself. "
                "You do NOT have access to executors. "
                "You do NOT grant permissions. "
                "You do NOT approve security requests. "
                "You only return one action for AgentCore to execute. "
                "\n\n"
                "Return exactly ONE JSON object. "
                "Do not return Markdown. "
                "Do not return code fences. "
                "Do not return explanations outside the JSON object. "
                "\n\n"
                "The JSON object MUST contain exactly these four fields: "
                "action_type, tool_id, inputs, reason. "
                "\n\n"
                "action_type MUST be exactly one of: "
                "INVOKE_TOOL, COMPLETE, FAIL. "
                "\n\n"
                "Rules for INVOKE_TOOL: "
                "tool_id must exactly match one of the IDs in "
                "available_tools. "
                "inputs must be a JSON object matching the tool's "
                "input_schema. "
                "\n\n"
                "Rules for COMPLETE: "
                "tool_id must be null. "
                "inputs must be null. "
                "\n\n"
                "Rules for FAIL: "
                "tool_id must be null. "
                "inputs must be null. "
                "\n\n"
                "Never invent a tool ID. "
                "Never use a tool that is not listed in "
                "available_tools. "
                "Never provide approval for a security request. "
                "\n\n"
                "For the current task, if the requested tool is "
                "available and has not yet been used, invoke it first. "
                "After receiving a successful tool result and when "
                "the task is complete, return COMPLETE."
            ),
        )

        user_message = LLMMessage(
            role="user",
            content=(
                "Current AgentContext:\n"
                + json.dumps(
                    context_data,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n\n"
                "Return exactly one valid action JSON object."
            ),
        )

        return LLMRequest(
            messages=(
                system_message,
                user_message,
            ),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @staticmethod
    def _serialize_tool_result(
        result: Any,
    ) -> dict[str, Any]:
        """
        Convert a ToolExecutionResult into safe LLM context.

        The result is represented as data only.
        """

        if result is None:
            return {
                "status": None,
                "summary": None,
                "artifacts": [],
            }

        if not isinstance(result, ToolExecutionResult):
            # Unrecognized result shape (e.g. a plain string or other
            # value recorded directly into AgentContext). Without this
            # fallback, getattr(..., default) silently returns the
            # defaults for every field and the actual value is dropped
            # entirely from the LLM's context -- the model would see
            # an empty-looking tool result and could re-invoke the same
            # tool or decide blind to what actually happened.
            return {
                "status": None,
                "summary": str(result),
                "artifacts": [],
            }

        status = getattr(
            result,
            "status",
            None,
        )

        summary = getattr(
            result,
            "summary",
            None,
        )

        artifacts = getattr(
            result,
            "artifacts",
            (),
        )

        if isinstance(
            artifacts,
            (tuple, list),
        ):
            safe_artifacts = [
                str(item)
                for item in artifacts
            ]
        else:
            safe_artifacts = [
                str(artifacts)
            ]

        return {
            "status": status,
            "summary": summary,
            "artifacts": safe_artifacts,
        }

    @staticmethod
    def _parse_action(
        content: str,
    ) -> AgentAction:
        """
        Parse the LLM response into AgentAction.

        The parser is intentionally strict about the action
        contract while tolerating harmless surrounding whitespace.
        """

        if not isinstance(
            content,
            str,
        ):
            raise TypeError(
                "LLM response content must be a string."
            )

        cleaned = content.strip()

        if not cleaned:
            raise ValueError(
                "LLM returned an empty response."
            )

        try:
            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "LLM response is not valid JSON."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "LLM response must be a JSON object."
            )

        allowed_fields = {
            "action_type",
            "tool_id",
            "inputs",
            "reason",
        }

        required_fields = {
            "action_type",
            "reason",
        }

        actual_fields = set(
            data.keys()
        )

        missing = required_fields - actual_fields
        extra = actual_fields - allowed_fields

        if missing or extra:

            details = []

            if missing:
                details.append(
                    f"missing={sorted(missing)}"
                )

            if extra:
                details.append(
                    f"extra={sorted(extra)}"
                )

            raise ValueError(
                "LLM response must contain action_type and reason, "
                "and may only additionally contain tool_id and "
                "inputs. "
                + " ".join(details)
            )

        action_type = data[
            "action_type"
        ]

        try:

            action_enum = AgentActionType(
                action_type
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "LLM returned an invalid action_type: "
                f"{action_type!r}."
            ) from exc

        tool_id = data.get(
            "tool_id"
        )

        inputs = data.get(
            "inputs"
        )

        reason = data[
            "reason"
        ]

        if not isinstance(
            reason,
            str,
        ):
            raise ValueError(
                "LLM action reason must be a string."
            )

        if action_enum == AgentActionType.INVOKE_TOOL:

            if not isinstance(
                tool_id,
                str,
            ) or not tool_id.strip():

                raise ValueError(
                    "INVOKE_TOOL requires a non-empty "
                    "tool_id."
                )

            if not isinstance(
                inputs,
                dict,
            ):

                raise ValueError(
                    "INVOKE_TOOL requires inputs to be "
                    "a JSON object."
                )

        elif action_enum in (
            AgentActionType.COMPLETE,
            AgentActionType.FAIL,
        ):

            if tool_id is not None:

                raise ValueError(
                    f"{action_enum.value} requires "
                    "tool_id to be null."
                )

            if inputs is not None:

                raise ValueError(
                    f"{action_enum.value} requires "
                    "inputs to be null."
                )

        return AgentAction(
            action_type=action_enum,
            tool_id=tool_id,
            inputs=inputs,
            reason=reason,
        )
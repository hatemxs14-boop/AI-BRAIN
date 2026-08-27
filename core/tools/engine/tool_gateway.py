from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.security.engine.authorization import Decision
from core.security.engine.security_decision import (
    SecurityDecision,
    SecurityDecisionPoint,
)

from core.tools.registry.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)

from core.tools.validation.input_validator import (
    InputValidationResult,
    InputValidator,
)

from core.tools.validation.output_validator import (
    OutputValidationResult,
    OutputValidator,
)


@dataclass(frozen=True)
class ToolExecutionResult:
    """
    Standard result returned by the Tool Gateway.
    """

    status: str
    summary: str
    next_actions: tuple[str, ...]
    artifacts: tuple[Any, ...]
    security_decision: SecurityDecision


class ToolGateway:
    """
    Security boundary between agents and executable tools.

    IMPORTANT ARCHITECTURAL RULE:

    ToolDefinition never contains an executor.

    The ToolRegistry exposes only public tool definitions.

    The ToolGateway owns the private executor mapping.

    Therefore:

        Agent
          |
          v
        Registry
          |
          v
    ToolDefinition
          |
          v
      ToolGateway
          |
          v
    Security Layer
          |
        ALLOW
          |
          v
    Private Executor
          |
          v
       Result

    The executor mapping is intentionally private to the Gateway.
    Agents must never receive it.
    """

    def __init__(
        self,
        security: SecurityDecisionPoint,
        registry: ToolRegistry,
    ) -> None:
        self.security = security
        self.registry = registry

        self.input_validator = InputValidator()
        self.output_validator = OutputValidator()

        # ---------------------------------------------------------
        # PRIVATE EXECUTOR REGISTRY
        #
        # Tool executors are deliberately NOT stored inside
        # ToolDefinition or ToolRegistry.
        #
        # Only the Gateway owns this mapping.
        # ---------------------------------------------------------

        self._executors: dict[str, Callable[..., Any]] = {}

    def register_executor(
        self,
        *,
        tool_id: str,
        executor: Callable[..., Any],
    ) -> None:
        """
        Register the private executor for a tool.

        The tool must already exist in the trusted ToolRegistry.

        Agents should never receive access to this method.
        """

        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError("tool_id must be a non-empty string.")

        if not callable(executor):
            raise TypeError("executor must be callable.")

        if not self.registry.contains(tool_id):
            raise KeyError(
                f"Cannot register executor for unknown tool: '{tool_id}'."
            )

        if tool_id in self._executors:
            raise ValueError(
                f"Executor for tool '{tool_id}' is already registered."
            )

        self._executors[tool_id] = executor

    def execute(
        self,
        *,
        subject: str,
        tool_id: str,
        tool_kwargs: dict[str, Any] | None = None,
        approved: bool | None = None,
        approved_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        """
        Execute a registered tool through the complete security boundary.

        Execution order:

            Registry
                ↓
            Executor Resolution
                ↓
            Input Validation
                ↓
            Security Layer
                ↓
            Approval
                ↓
            Private Executor
                ↓
            Output Validation
                ↓
            Final Result
        """

        # ---------------------------------------------------------
        # 1. Resolve public tool definition.
        # ---------------------------------------------------------

        try:
            tool: ToolDefinition = self.registry.get(tool_id)

        except (KeyError, TypeError) as exc:
            return ToolExecutionResult(
                status="DENIED",
                summary="Tool is not registered in the Tool Registry.",
                next_actions=(
                    "Do not execute the tool.",
                    "Register the tool before requesting execution.",
                ),
                artifacts=(str(exc),),
                security_decision=self._unknown_tool_security_decision(
                    subject=subject,
                    tool_id=tool_id,
                ),
            )

        # ---------------------------------------------------------
        # 2. Resolve PRIVATE executor.
        #
        # The executor is not obtained from ToolDefinition.
        # ---------------------------------------------------------

        executor = self._executors.get(tool_id)

        if executor is None:
            return ToolExecutionResult(
                status="DENIED",
                summary="Tool has no registered private executor.",
                next_actions=(
                    "Do not execute the tool.",
                    "Register the tool executor with the Tool Gateway.",
                ),
                artifacts=(
                    f"No private executor registered for '{tool_id}'.",
                ),
                security_decision=self._security_for_registered_tool(
                    subject=subject,
                    tool=tool,
                ),
            )

        # ---------------------------------------------------------
        # 3. Normalize inputs.
        # ---------------------------------------------------------

        if tool_kwargs is None:
            tool_kwargs = {}

        if not isinstance(tool_kwargs, dict):
            return ToolExecutionResult(
                status="INVALID_INPUT",
                summary="Tool input validation failed.",
                next_actions=(
                    "Do not execute the tool.",
                    "Provide tool inputs as a dictionary.",
                    "Re-submit the request after validation succeeds.",
                ),
                artifacts=("tool_kwargs must be a dictionary.",),
                security_decision=self._security_for_registered_tool(
                    subject=subject,
                    tool=tool,
                ),
            )

        # ---------------------------------------------------------
        # 4. Validate inputs.
        # ---------------------------------------------------------

        validation: InputValidationResult = (
            self.input_validator.validate(
                input_schema=tool.input_schema,
                inputs=tool_kwargs,
            )
        )

        if not validation.valid:
            return ToolExecutionResult(
                status="INVALID_INPUT",
                summary=validation.summary,
                next_actions=(
                    "Do not execute the tool.",
                    "Correct the tool inputs.",
                    "Re-submit the request after validation succeeds.",
                ),
                artifacts=validation.errors,
                security_decision=self._security_for_registered_tool(
                    subject=subject,
                    tool=tool,
                ),
            )

        # ---------------------------------------------------------
        # 5. Use trusted ToolDefinition security contract.
        # ---------------------------------------------------------

        resource = tool.resource
        action = tool.action
        scope = tool.scope

        # ---------------------------------------------------------
        # 6. Security evaluation.
        # ---------------------------------------------------------

        if approved is None:
            security_decision = self.security.evaluate(
                subject=subject,
                resource=resource,
                action=action,
                scope=scope,
                metadata=metadata,
            )
        else:
            security_decision = self.security.evaluate_with_approval(
                subject=subject,
                resource=resource,
                action=action,
                scope=scope,
                approved=approved,
                approved_by=approved_by,
                metadata=metadata,
            )

        # ---------------------------------------------------------
        # 7. Verify risk consistency.
        #
        # Compared against the EFFECTIVE risk the Security Layer
        # actually decided on (security_decision.risk.level is the
        # raw, pre-permission-floor RiskEngine assessment -- comparing
        # against that instead reintroduces the exact "conservative
        # permission becomes permanently unauthorizable" bug that was
        # already fixed in AuthorizationEngine.authorize(), just one
        # layer up: a tool honestly declaring a higher risk_level than
        # RiskEngine's keyword heuristic would guess was being denied
        # here unconditionally, even with valid policy and explicit
        # human approval).
        # ---------------------------------------------------------

        registered_risk = tool.risk_level
        assessed_risk = security_decision.authorization.effective_risk

        if not self._risk_is_consistent(
            registered_risk=registered_risk,
            assessed_risk=assessed_risk,
        ):
            return ToolExecutionResult(
                status="DENIED",
                summary=(
                    "Tool risk contract is inconsistent with "
                    "the Security Layer assessment."
                ),
                next_actions=(
                    "Do not execute the tool.",
                    "Review the registered tool risk level.",
                    "Review the Security Layer risk assessment.",
                ),
                artifacts=(
                    f"registered_risk={registered_risk}",
                    f"assessed_risk={assessed_risk}",
                ),
                security_decision=security_decision,
            )

        # ---------------------------------------------------------
        # 8. DENY.
        # ---------------------------------------------------------

        if security_decision.decision == Decision.DENY:
            return ToolExecutionResult(
                status="DENIED",
                summary="Tool execution denied by the Security Layer.",
                next_actions=(
                    "Do not execute the tool.",
                    "Review the authorization and security decision.",
                ),
                artifacts=(),
                security_decision=security_decision,
            )

        # ---------------------------------------------------------
        # 9. REQUIRE APPROVAL.
        # ---------------------------------------------------------

        if security_decision.decision == Decision.REQUIRE_APPROVAL:
            return ToolExecutionResult(
                status="APPROVAL_REQUIRED",
                summary="Tool execution requires additional approval.",
                next_actions=(
                    "Obtain the required approval.",
                    "Re-submit the request with the explicit approval result.",
                ),
                artifacts=(),
                security_decision=security_decision,
            )

        # ---------------------------------------------------------
        # 10. Fail closed on unknown decisions.
        #
        # ALLOW_WITH_CONTROLS is a real, distinct decision
        # AuthorizationEngine can return for MEDIUM-risk operations
        # (see authorization.py's `_decision_for_risk`) -- it means
        # "execute automatically, within controls", not "unknown".
        # SecurityDecisionPoint now preserves it instead of silently
        # rewriting it to plain ALLOW, so this check must accept it
        # too; otherwise every MEDIUM-risk tool call would newly and
        # incorrectly fail closed here.
        # ---------------------------------------------------------

        if security_decision.decision not in (
            Decision.ALLOW,
            Decision.ALLOW_WITH_CONTROLS,
        ):
            return ToolExecutionResult(
                status="DENIED",
                summary="Unknown security decision; execution blocked.",
                next_actions=(
                    "Inspect the Security Layer decision.",
                    "Fail closed.",
                ),
                artifacts=(),
                security_decision=security_decision,
            )

        # ---------------------------------------------------------
        # 11. Execute ONLY the private executor.
        #
        # At this point:
        #
        # - tool exists
        # - inputs are valid
        # - risk is consistent
        # - authorization succeeded
        # - approval requirements are satisfied
        # - security decision is ALLOW
        # ---------------------------------------------------------

        try:
            output = executor(**tool_kwargs)

        except Exception as exc:
            return ToolExecutionResult(
                status="ERROR",
                summary="Authorized tool execution failed.",
                next_actions=(
                    "Inspect the tool execution error.",
                    "Do not silently retry an unknown failure.",
                ),
                artifacts=(str(exc),),
                security_decision=security_decision,
            )

        # ---------------------------------------------------------
        # 12. Validate output.
        # ---------------------------------------------------------

        output_validation: OutputValidationResult = (
            self.output_validator.validate(
                output_schema=tool.output_schema,
                output=output,
            )
        )

        if not output_validation.valid:
            return ToolExecutionResult(
                status="INVALID_OUTPUT",
                summary=output_validation.summary,
                next_actions=(
                    "Do not trust the tool output.",
                    "Inspect the tool output against its registered schema.",
                    "Review the tool implementation.",
                ),
                artifacts=output_validation.errors,
                security_decision=security_decision,
            )

        # ---------------------------------------------------------
        # 13. Successful execution.
        # ---------------------------------------------------------

        return ToolExecutionResult(
            status="SUCCESS",
            summary="Tool executed successfully.",
            next_actions=(),
            artifacts=(output,),
            security_decision=security_decision,
        )

    @staticmethod
    def _risk_is_consistent(
        *,
        registered_risk: str,
        assessed_risk: str,
    ) -> bool:
        """
        Return True when the Security Layer assessment is equal to
        or stricter than the registered Tool risk.
        """

        levels = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

        registered_value = levels.get(registered_risk)
        assessed_value = levels.get(assessed_risk)

        if registered_value is None or assessed_value is None:
            return False

        return assessed_value >= registered_value

    def _security_for_registered_tool(
        self,
        *,
        subject: str,
        tool: ToolDefinition,
    ) -> SecurityDecision:
        """
        Evaluate security for a registered tool.
        """

        return self.security.evaluate(
            subject=subject,
            resource=tool.resource,
            action=tool.action,
            scope=tool.scope,
        )

    def _unknown_tool_security_decision(
        self,
        *,
        subject: str,
        tool_id: str,
    ) -> SecurityDecision:
        """
        Generate a real Security Layer denial for an unknown tool.
        """

        return self.security.evaluate(
            subject=subject,
            resource=tool_id,
            action="execute",
            scope="unknown",
        )
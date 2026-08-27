from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutputValidationResult:
    valid: bool
    summary: str
    errors: tuple[str, ...]


class OutputValidator:
    """
    Validates tool outputs against the registered output schema.

    This validator does not execute tools.
    It only validates the structure and basic type of outputs.
    """

    def validate(
        self,
        *,
        output_schema: dict[str, Any],
        output: Any,
    ) -> OutputValidationResult:

        errors: list[str] = []

        if not isinstance(output_schema, dict):
            return OutputValidationResult(
                valid=False,
                summary="Tool output validation failed.",
                errors=("output_schema must be a dictionary.",),
            )

        expected_type = output_schema.get("type")

        if expected_type is None:
            return OutputValidationResult(
                valid=True,
                summary="Tool output validation succeeded.",
                errors=(),
            )

        if not self._matches_type(
            value=output,
            expected_type=expected_type,
        ):
            errors.append(
                f"Invalid output type: expected '{expected_type}'."
            )

        if errors:
            return OutputValidationResult(
                valid=False,
                summary="Tool output validation failed.",
                errors=tuple(errors),
            )

        return OutputValidationResult(
            valid=True,
            summary="Tool output validation succeeded.",
            errors=(),
        )

    @staticmethod
    def _matches_type(
        *,
        value: Any,
        expected_type: str,
    ) -> bool:

        if expected_type == "string":
            return isinstance(value, str)

        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)

        if expected_type == "number":
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        if expected_type == "boolean":
            return isinstance(value, bool)

        if expected_type == "object":
            return isinstance(value, dict)

        if expected_type == "array":
            return isinstance(value, list)

        if expected_type == "null":
            return value is None

        return False
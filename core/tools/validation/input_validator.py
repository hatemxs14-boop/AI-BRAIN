from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InputValidationResult:
    valid: bool
    summary: str
    errors: tuple[str, ...]


class InputValidator:
    """
    Validates tool inputs against the registered input schema.

    This validator does not execute tools.
    It only validates the structure and basic types of inputs.
    """

    def validate(
        self,
        *,
        input_schema: dict[str, Any],
        inputs: dict[str, Any],
    ) -> InputValidationResult:

        errors: list[str] = []

        if not isinstance(input_schema, dict):
            return InputValidationResult(
                valid=False,
                summary="Tool input validation failed.",
                errors=("input_schema must be a dictionary.",),
            )

        if not isinstance(inputs, dict):
            return InputValidationResult(
                valid=False,
                summary="Tool input validation failed.",
                errors=("Tool inputs must be a dictionary.",),
            )

        properties = input_schema.get("properties", {})

        if not isinstance(properties, dict):
            errors.append(
                "input_schema.properties must be a dictionary."
            )
            properties = {}

        required = input_schema.get("required", [])

        if not isinstance(required, list):
            errors.append(
                "input_schema.required must be a list."
            )
            required = []

        # ---------------------------------------------------------
        # Required fields
        # ---------------------------------------------------------

        for field in required:
            if field not in inputs:
                errors.append(
                    f"Missing required input field: '{field}'."
                )

        # ---------------------------------------------------------
        # Unknown fields
        # ---------------------------------------------------------

        additional_properties = input_schema.get(
            "additionalProperties",
            True,
        )

        if additional_properties is False:
            for field in inputs:
                if field not in properties:
                    errors.append(
                        f"Unknown input field: '{field}'."
                    )

        # ---------------------------------------------------------
        # Type validation
        # ---------------------------------------------------------

        for field, value in inputs.items():

            if field not in properties:
                continue

            field_schema = properties[field]

            if not isinstance(field_schema, dict):
                errors.append(
                    f"Invalid schema definition for field '{field}'."
                )
                continue

            expected_type = field_schema.get("type")

            if expected_type is None:
                continue

            if not self._matches_type(
                value=value,
                expected_type=expected_type,
            ):
                errors.append(
                    f"Invalid type for field '{field}': "
                    f"expected '{expected_type}'."
                )

        if errors:
            return InputValidationResult(
                valid=False,
                summary="Tool input validation failed.",
                errors=tuple(errors),
            )

        return InputValidationResult(
            valid=True,
            summary="Tool input validation succeeded.",
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
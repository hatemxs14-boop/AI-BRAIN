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

    Validation recurses into nested "object" (properties/required) and
    "array" (items) schemas, not just the top-level type -- otherwise
    an executor could return an object missing every field its schema
    requires (or carrying an unexpected nested value) and still have
    it reported as SUCCESS, since only `isinstance(output, dict)` was
    ever checked.
    """

    def validate(
        self,
        *,
        output_schema: dict[str, Any],
        output: Any,
    ) -> OutputValidationResult:

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

        errors = self._validate_value(
            schema=output_schema,
            value=output,
            path="output",
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

    def _validate_value(
        self,
        *,
        schema: dict[str, Any],
        value: Any,
        path: str,
    ) -> list[str]:

        errors: list[str] = []

        expected_type = schema.get("type")

        if expected_type is not None and not self._matches_type(
            value=value,
            expected_type=expected_type,
        ):
            label = (
                "Invalid output type"
                if path == "output"
                else f"Invalid type for output field '{path}'"
            )
            errors.append(f"{label}: expected '{expected_type}'.")
            # The value doesn't even match the declared shape --
            # recursing into it would only produce confusing,
            # redundant nested errors.
            return errors

        if (
            self._type_includes(expected_type, "object")
            and isinstance(value, dict)
        ):
            properties = schema.get("properties", {})

            if isinstance(properties, dict):

                required = schema.get("required", [])

                if isinstance(required, list):
                    for field in required:
                        if field not in value:
                            errors.append(
                                "Missing required output field: "
                                f"'{path}.{field}'."
                            )

                for field, field_value in value.items():

                    if field not in properties:
                        continue

                    field_schema = properties[field]

                    if not isinstance(field_schema, dict):
                        errors.append(
                            "Invalid schema definition for output "
                            f"field '{path}.{field}'."
                        )
                        continue

                    errors.extend(
                        self._validate_value(
                            schema=field_schema,
                            value=field_value,
                            path=f"{path}.{field}",
                        )
                    )

        if (
            self._type_includes(expected_type, "array")
            and isinstance(value, list)
        ):
            items_schema = schema.get("items")

            if isinstance(items_schema, dict):
                for index, item in enumerate(value):
                    errors.extend(
                        self._validate_value(
                            schema=items_schema,
                            value=item,
                            path=f"{path}[{index}]",
                        )
                    )

        return errors

    @staticmethod
    def _type_includes(expected_type: Any, type_name: str) -> bool:
        if expected_type is None:
            return False

        if isinstance(expected_type, list):
            return type_name in expected_type

        return expected_type == type_name

    @staticmethod
    def _matches_type(
        *,
        value: Any,
        expected_type: Any,
    ) -> bool:

        # JSON Schema allows "type" to be a list of allowed types
        # (e.g. ["string", "null"] for a nullable field).
        if isinstance(expected_type, list):
            return any(
                OutputValidator._matches_type(
                    value=value,
                    expected_type=single_type,
                )
                for single_type in expected_type
            )

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

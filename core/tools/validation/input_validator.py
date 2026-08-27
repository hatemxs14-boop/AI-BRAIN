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

    Validation recurses into nested "object" (properties/required) and
    "array" (items) schemas, not just the top-level field shape --
    otherwise a field declared e.g. {"type": "array", "items":
    {"type": "string"}} would only ever be checked for "is this a
    list", and semantically invalid nested values (wrong element
    types, missing nested required fields) would reach the executor
    unchecked.
    """

    def validate(
        self,
        *,
        input_schema: dict[str, Any],
        inputs: dict[str, Any],
    ) -> InputValidationResult:

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

        errors = self._validate_object_fields(
            schema=input_schema,
            value=inputs,
            path="",
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

    def _validate_object_fields(
        self,
        *,
        schema: dict[str, Any],
        value: dict[str, Any],
        path: str,
    ) -> list[str]:
        """
        Validate `value` (a dict) against an object-like schema's
        `properties`/`required`/`additionalProperties`, recursing into
        nested field schemas. `path` is a dotted prefix used only for
        nested error messages -- the top-level call uses an empty
        path, so existing top-level error text is unchanged.
        """

        errors: list[str] = []

        properties = schema.get("properties", {})

        if not isinstance(properties, dict):
            errors.append(
                "input_schema.properties must be a dictionary."
            )
            properties = {}

        required = schema.get("required", [])

        if not isinstance(required, list):
            errors.append(
                "input_schema.required must be a list."
            )
            required = []

        # ---------------------------------------------------------
        # Required fields
        # ---------------------------------------------------------

        for field in required:
            if field not in value:
                errors.append(
                    "Missing required input field: "
                    f"'{self._field_name(path, field)}'."
                )

        # ---------------------------------------------------------
        # Unknown fields
        # ---------------------------------------------------------

        additional_properties = schema.get(
            "additionalProperties",
            True,
        )

        if additional_properties is False:
            for field in value:
                if field not in properties:
                    errors.append(
                        "Unknown input field: "
                        f"'{self._field_name(path, field)}'."
                    )

        # ---------------------------------------------------------
        # Type validation (recursing into nested object/array shapes)
        # ---------------------------------------------------------

        for field, field_value in value.items():

            if field not in properties:
                continue

            field_schema = properties[field]

            if not isinstance(field_schema, dict):
                errors.append(
                    "Invalid schema definition for field "
                    f"'{self._field_name(path, field)}'."
                )
                continue

            errors.extend(
                self._validate_field(
                    schema=field_schema,
                    value=field_value,
                    field_name=self._field_name(path, field),
                )
            )

        return errors

    def _validate_field(
        self,
        *,
        schema: dict[str, Any],
        value: Any,
        field_name: str,
    ) -> list[str]:

        errors: list[str] = []

        expected_type = schema.get("type")

        if expected_type is not None:

            if not self._matches_type(
                value=value,
                expected_type=expected_type,
            ):
                errors.append(
                    f"Invalid type for field '{field_name}': "
                    f"expected '{expected_type}'."
                )
                # The value doesn't even match the declared shape --
                # recursing into it would only produce confusing,
                # redundant nested errors.
                return errors

        if (
            self._type_includes(expected_type, "object")
            and isinstance(value, dict)
        ):
            errors.extend(
                self._validate_object_fields(
                    schema=schema,
                    value=value,
                    path=field_name,
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
                        self._validate_field(
                            schema=items_schema,
                            value=item,
                            field_name=f"{field_name}[{index}]",
                        )
                    )

        return errors

    @staticmethod
    def _field_name(path: str, field: str) -> str:
        return f"{path}.{field}" if path else field

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
                InputValidator._matches_type(
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

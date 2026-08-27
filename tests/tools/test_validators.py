"""
Regression tests for InputValidator/OutputValidator's recursive
validation of nested "object" (properties/required) and "array"
(items) schemas.

Both validators used to check only the top-level shape: a field
declared {"type": "array", "items": {"type": "string"}} was only ever
checked for "is this a list", and nested object properties/required
were never recursed into -- letting semantically invalid inputs reach
executors, and malformed/incomplete outputs reach callers as SUCCESS.
"""
from __future__ import annotations

from core.tools.validation.input_validator import InputValidator
from core.tools.validation.output_validator import OutputValidator


# ---------------------------------------------------------------------
# InputValidator
# ---------------------------------------------------------------------

def test_input_validator_rejects_wrong_typed_array_items():
    validator = InputValidator()

    result = validator.validate(
        input_schema={
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["tags"],
        },
        inputs={"tags": [1, 2, "ok"]},
    )

    assert result.valid is False
    assert any("tags[0]" in error for error in result.errors)


def test_input_validator_accepts_correctly_typed_array_items():
    validator = InputValidator()

    result = validator.validate(
        input_schema={
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["tags"],
        },
        inputs={"tags": ["a", "b", "c"]},
    )

    assert result.valid is True


def test_input_validator_recurses_into_nested_object_required_fields():
    validator = InputValidator()

    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["city", "zip"],
            }
        },
        "required": ["address"],
    }

    result = validator.validate(
        input_schema=schema,
        inputs={"address": {"city": "Cairo"}},
    )

    assert result.valid is False
    assert any("address.zip" in error for error in result.errors)

    ok = validator.validate(
        input_schema=schema,
        inputs={"address": {"city": "Cairo", "zip": "12345"}},
    )
    assert ok.valid is True


def test_input_validator_supports_list_of_types():
    """
    JSON Schema allows "type" to be a list of allowed types (e.g. a
    nullable field). The old single-string-only check rejected every
    value for such a field, including valid ones.
    """

    validator = InputValidator()

    schema = {
        "type": "object",
        "properties": {
            "nickname": {"type": ["string", "null"]},
        },
        "required": [],
    }

    assert validator.validate(
        input_schema=schema,
        inputs={"nickname": "Bob"},
    ).valid is True

    assert validator.validate(
        input_schema=schema,
        inputs={"nickname": None},
    ).valid is True

    assert validator.validate(
        input_schema=schema,
        inputs={"nickname": 123},
    ).valid is False


# ---------------------------------------------------------------------
# OutputValidator
# ---------------------------------------------------------------------

def test_output_validator_rejects_missing_nested_required_field():
    validator = OutputValidator()

    result = validator.validate(
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
        output={"status": "ok"},
    )

    assert result.valid is False
    assert any("summary" in error for error in result.errors)


def test_output_validator_accepts_valid_nested_object():
    validator = OutputValidator()

    result = validator.validate(
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
        output={"status": "ok", "summary": "done"},
    )

    assert result.valid is True


def test_output_validator_rejects_wrong_typed_array_items():
    validator = OutputValidator()

    result = validator.validate(
        output_schema={
            "type": "array",
            "items": {"type": "string"},
        },
        output=["a", 2, "c"],
    )

    assert result.valid is False

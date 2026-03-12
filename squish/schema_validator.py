"""squish/schema_validator.py

SchemaValidator — JSON schema validator for structured generation outputs.

Modern LLM applications increasingly rely on *structured generation* — having
the model emit JSON that conforms to a caller-defined schema (function calls,
tool use, data extraction pipelines).  Validating the output before passing it
downstream prevents cascading failures and enables fast, informative error
messages that can be fed back as correction prompts.

SchemaValidator implements a practical subset of JSON Schema sufficient for
the vast majority of structured-generation use cases:

* ``type`` — maps to Python built-in types (``string``, ``number``,
  ``integer``, ``boolean``, ``array``, ``object``, ``null``).
* ``required`` — checks that all listed property names are present in an
  object.
* ``properties`` — recursively validates each named property value.
* ``minLength`` / ``maxLength`` — enforces string length bounds.
* ``minimum`` / ``maximum`` — enforces numeric bounds (inclusive).
* ``items`` — validates every element of an array against a sub-schema.

No external dependencies are required; only Python's built-in ``json`` module
is used.

Example usage::

    from squish.schema_validator import SchemaValidator

    validator = SchemaValidator()
    schema = {
        "type": "object",
        "required": ["name", "score"],
        "properties": {
            "name":  {"type": "string", "minLength": 1},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }
    result = validator.validate('{"name": "Alice", "score": 0.95}', schema)
    print(result.valid, result.errors)
"""

from __future__ import annotations

__all__ = ["ValidationResult", "SchemaValidator"]

import json
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


# Maps JSON Schema type names to the corresponding Python types.
_TYPE_MAP: dict[str, tuple] = {
    "string":  (str,),
    "number":  (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array":   (list,),
    "object":  (dict,),
    "null":    (type(None),),
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Output of :meth:`SchemaValidator.validate`.

    Attributes:
        valid:            ``True`` iff no validation errors were found.
        errors:           Human-readable error messages; empty on success.
        n_fields_checked: Total number of schema constraints evaluated
                          across the entire document tree.
    """

    valid: bool
    errors: list[str]
    n_fields_checked: int


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """JSON schema validator for structured LLM generation outputs.

    Supports a practical subset of JSON Schema:
    ``type``, ``required``, ``properties``, ``minLength``, ``maxLength``,
    ``minimum``, ``maximum``, and ``items``.

    Thread-safe for concurrent use (the instance carries no mutable state).
    """

    def validate(self, json_str: str, schema: dict) -> ValidationResult:
        """Validate *json_str* against *schema*.

        Args:
            json_str: Raw JSON string produced by the model.
            schema:   Schema dict following the supported subset of JSON
                      Schema draft-07.

        Returns:
            A :class:`ValidationResult` with a list of errors and the
            cumulative number of schema constraints checked.
        """
        try:
            value = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return ValidationResult(
                valid=False,
                errors=[f"JSON parse error: {exc}"],
                n_fields_checked=0,
            )

        counter: list[int] = [0]
        errors = self._check(value, schema, "", counter)
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            n_fields_checked=counter[0],
        )

    def validate_value(
        self,
        value: object,
        schema: dict,
        path: str = "",
    ) -> list[str]:
        """Recursively validate *value* against *schema*.

        Args:
            value:  A parsed Python value (``str``, ``int``, ``float``,
                    ``bool``, ``list``, ``dict``, or ``None``).
            schema: Schema dict.
            path:   Dot-path prefix used in error messages
                    (e.g. ``"root.items[0]"``).

        Returns:
            List of error strings.  An empty list means the value is valid.
        """
        return self._check(value, schema, path, [0])

    def is_valid(self, json_str: str, schema: dict) -> bool:
        """Return ``True`` iff *json_str* is valid against *schema*.

        Args:
            json_str: Raw JSON string.
            schema:   Schema dict.

        Returns:
            ``True`` on validation success, ``False`` otherwise.
        """
        return self.validate(json_str, schema).valid

    # ------------------------------------------------------------------
    # Internal recursive implementation
    # ------------------------------------------------------------------

    def _check(
        self,
        value: object,
        schema: dict,
        path: str,
        counter: list[int],
    ) -> list[str]:
        """Core recursive validation.

        *counter* is a single-element list used as a mutable integer so
        that nested recursive calls accumulate into the same total.

        Args:
            value:   Parsed Python value.
            schema:  Schema dict.
            path:    Dot-path string for error messages.
            counter: ``[n]`` — incremented once per constraint checked.

        Returns:
            List of validation error strings for this node and all
            descendants.
        """
        errors: list[str] = []
        prefix = f"{path}: " if path else ""

        # ── type ──────────────────────────────────────────────────────
        if "type" in schema:
            counter[0] += 1
            type_name = schema["type"]
            expected = _TYPE_MAP.get(type_name)
            if expected is None:
                errors.append(
                    f"{prefix}unknown schema type {type_name!r}"
                )
            else:
                # In Python bool is a subclass of int; guard against
                # false positives when the target type is integer/number.
                if type_name in ("integer", "number") and isinstance(value, bool):
                    errors.append(
                        f"{prefix}expected {type_name}, got boolean"
                    )
                elif not isinstance(value, expected):
                    errors.append(
                        f"{prefix}expected type {type_name!r}, "
                        f"got {type(value).__name__!r}"
                    )

        # ── string constraints ─────────────────────────────────────────
        if isinstance(value, str):
            if "minLength" in schema:
                counter[0] += 1
                if len(value) < schema["minLength"]:
                    errors.append(
                        f"{prefix}string length {len(value)} < "
                        f"minLength {schema['minLength']}"
                    )
            if "maxLength" in schema:
                counter[0] += 1
                if len(value) > schema["maxLength"]:
                    errors.append(
                        f"{prefix}string length {len(value)} > "
                        f"maxLength {schema['maxLength']}"
                    )

        # ── numeric constraints ────────────────────────────────────────
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema:
                counter[0] += 1
                if value < schema["minimum"]:
                    errors.append(
                        f"{prefix}value {value} < minimum {schema['minimum']}"
                    )
            if "maximum" in schema:
                counter[0] += 1
                if value > schema["maximum"]:
                    errors.append(
                        f"{prefix}value {value} > maximum {schema['maximum']}"
                    )

        # ── object constraints ─────────────────────────────────────────
        if isinstance(value, dict):
            if "required" in schema:
                for req_key in schema["required"]:
                    counter[0] += 1
                    if req_key not in value:
                        errors.append(
                            f"{prefix}missing required field {req_key!r}"
                        )
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    if prop_name in value:
                        child_path = (
                            f"{path}.{prop_name}" if path else prop_name
                        )
                        errors.extend(
                            self._check(value[prop_name], prop_schema, child_path, counter)
                        )

        # ── array constraints ──────────────────────────────────────────
        if isinstance(value, list) and "items" in schema:
            for idx, item in enumerate(value):
                item_path = f"{path}[{idx}]"
                errors.extend(
                    self._check(item, schema["items"], item_path, counter)
                )

        return errors

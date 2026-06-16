"""Load and validate against the MSG operator JSON schema."""

import json
import pathlib

import jsonschema


def load_schema() -> dict:
    ref = pathlib.Path(__file__).parents[2] / "data" / "msg_operators.schema.json"
    with ref.open() as f:
        return json.load(f)


def validate(data: dict) -> None:
    """Raise jsonschema.ValidationError if data does not conform to schema."""
    schema = load_schema()
    jsonschema.validate(instance=data, schema=schema)

"""Tests that the schema itself is well-formed and that a minimal valid
document passes validation."""

import pytest
import jsonschema

from msgjson.schema import load_schema, validate


def minimal_doc(uni=1, sg=1, type_=1):
    return {
        "version": "1.0",
        "source": "test",
        "groups": [
            {
                "uni_number": uni,
                "bns_number": "1.1",
                "og_number": "1.1.1",
                "type": type_,
                "parent_sg": sg,
                "operators": [
                    {
                        "W": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "t": [0.0, 0.0, 0.0],
                        "theta": 1,
                    }
                ],
            }
        ],
    }


def test_schema_loads():
    schema = load_schema()
    assert "$schema" in schema


def test_valid_minimal_document():
    validate(minimal_doc())


def test_invalid_theta():
    doc = minimal_doc()
    doc["groups"][0]["operators"][0]["theta"] = 0
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_invalid_type():
    doc = minimal_doc()
    doc["groups"][0]["type"] = 5
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_missing_operators():
    doc = minimal_doc()
    del doc["groups"][0]["operators"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_w_must_be_integer():
    doc = minimal_doc()
    doc["groups"][0]["operators"][0]["W"][0][0] = 0.5
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)

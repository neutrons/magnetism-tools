"""Validate an existing msg_operators.json file against the schema."""

import argparse
import json
import pathlib
import sys

from .schema import validate, load_schema


DEFAULT_FILE = pathlib.Path(__file__).parents[2] / "data" / "msg_operators.json"


def validate_file(path: pathlib.Path) -> bool:
    with path.open() as f:
        data = json.load(f)
    try:
        validate(data)
        print(f"OK – {path} is valid ({len(data['groups'])} groups)")
        return True
    except Exception as e:
        print(f"INVALID – {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate MSG operator JSON")
    parser.add_argument(
        "file",
        nargs="?",
        type=pathlib.Path,
        default=DEFAULT_FILE,
    )
    args = parser.parse_args()
    sys.exit(0 if validate_file(args.file) else 1)

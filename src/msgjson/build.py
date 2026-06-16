"""Build data/msg_operators.json from spglib.

Usage:
    python -m msgjson.build
    python -m msgjson.build --output path/to/output.json
"""

import argparse
import json
import pathlib

import spglib

from .spglib_source import iter_all_msgs
from .schema import validate


DEFAULT_OUTPUT = pathlib.Path(__file__).parents[2] / "data" / "msg_operators.json"


def build_table() -> dict:
    groups = []
    for meta, operators in iter_all_msgs():
        entry = {**meta, "operators": operators}
        groups.append(entry)

    return {
        "version": "1.0",
        "source": f"spglib {spglib.__version__}",
        "groups": groups,
    }


def main(output: pathlib.Path = DEFAULT_OUTPUT) -> None:
    print("Building MSG operator table …")
    data = build_table()

    print("Validating against schema …")
    validate(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Written {len(data['groups'])} groups → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MSG operator JSON table")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the JSON file",
    )
    args = parser.parse_args()
    main(args.output)

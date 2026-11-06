#!/usr/bin/env python3
"""Export a reviewed structure map as a staging-upload template."""

import argparse
import json
from pathlib import Path

from app.contracts import StructureTemplate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed_map", type=Path)
    parser.add_argument("--document-title", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reviewed = json.loads(args.reviewed_map.read_text(encoding="utf-8"))
    if not isinstance(reviewed, dict) or reviewed.get("schema_version") != "reviewed-structure-map-v1":
        raise ValueError("expected a reviewed structure-map envelope")
    structure_map = reviewed.get("structure_map")
    if not isinstance(structure_map, dict) or not args.document_title.strip():
        raise ValueError("structure map entries and document title are required")
    template = StructureTemplate.model_validate({
        "schema_version": "structure-template-v1",
        "document_title": args.document_title.strip(),
        "instructions": ["Review the document title and page ranges before uploading with the PDF."],
        "entries": structure_map.get("entries"),
    })
    with args.output.open("x", encoding="utf-8") as output:
        output.write(template.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

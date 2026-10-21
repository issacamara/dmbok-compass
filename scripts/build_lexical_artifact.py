#!/usr/bin/env python3
"""Small CLI entry point for building a lexical artifact from JSON chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.corpus.chunking import TextChunk
from app.corpus.lexical_artifact import build_lexical_artifact


class LocalObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, name: str, data: bytes, *, metadata: dict[str, str], if_absent: bool = True) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if if_absent and target.exists():
            raise FileExistsError(name)
        target.write_bytes(data)
        target.with_suffix(target.suffix + ".metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    def get(self, name: str) -> bytes:
        return (self.root / name).read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus-version-id", required=True)
    parser.add_argument("--object-name", default="lexical/artifact.json")
    args = parser.parse_args()
    chunks = tuple(TextChunk(**item) for item in json.loads(args.chunks.read_text(encoding="utf-8")))
    manifest = build_lexical_artifact(
        chunks, corpus_version_id=args.corpus_version_id, object_name=args.object_name, store=LocalObjectStore(args.output)
    )
    print(json.dumps(asdict(manifest), sort_keys=True))


if __name__ == "__main__":
    main()

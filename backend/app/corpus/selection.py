"""Resolve the corpus selected by the upload/deployment workflow."""

from __future__ import annotations

import os

from fastapi import HTTPException, status


def uploaded_corpus_version_id() -> str:
    """Keep corpus identity internal to the evaluation UI."""

    version_id = os.environ.get("RAG_CORPUS_VERSION_ID", "").strip()
    if not version_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "uploaded_corpus_unavailable", "message": "The uploaded corpus is not configured."},
        )
    return version_id

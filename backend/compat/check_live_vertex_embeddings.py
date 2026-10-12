"""Verify the configured Vertex embedding route with native LangChain calls."""

from __future__ import annotations

import math
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def main() -> None:
    project = os.environ["DMBOK_COMPAT_PROJECT_ID"]
    location = os.environ["DMBOK_COMPAT_VERTEX_LOCATION"]
    if not project.strip() or not location.strip():
        raise ValueError("Explicit development project and Vertex location are required")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        vertexai=True,
        project=project,
        location=location,
        output_dimensionality=768,
    )
    document = embeddings.embed_documents(["An authored document for compatibility testing."])[0]
    query = embeddings.embed_query("An authored compatibility question.")
    assert len(document) == len(query) == 768
    assert all(math.isfinite(value) for value in document + query)
    print("Live Vertex document/query embeddings at 768 dimensions: passed")


if __name__ == "__main__":
    main()

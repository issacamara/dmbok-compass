"""Exercise the patched native vector store against an indexed development collection.

Set DMBOK_COMPAT_PROJECT_ID and DMBOK_COMPAT_COLLECTION explicitly. The
collection must have a 768-dimensional cosine index and a composite index for
metadata.corpus_version_id, as specified by the target architecture.
"""

from __future__ import annotations

import math
import os
from uuid import uuid4

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_firestore import FirestoreVectorStore


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector("near")

    @staticmethod
    def _vector(text: str) -> list[float]:
        return ([1.0, 0.0] if text == "near" else [0.0, 1.0]) + [0.0] * 766


def main() -> None:
    project = os.environ["DMBOK_COMPAT_PROJECT_ID"]
    collection_name = os.environ["DMBOK_COMPAT_COLLECTION"]
    if not project.strip() or not collection_name.strip():
        raise ValueError("Explicit development project and indexed collection are required")

    client = firestore.Client(project=project)
    collection = client.collection(collection_name)
    store = FirestoreVectorStore(
        collection, DeterministicEmbeddings(), client=client, content_field="text"
    )
    marker = "compat-" + uuid4().hex
    ids = [f"{marker}-near", f"{marker}-far", f"{marker}-excluded"]
    documents = [
        Document(page_content="near", metadata={"corpus_version_id": marker, "page": 1}),
        Document(page_content="far", metadata={"corpus_version_id": marker, "page": 2}),
        Document(page_content="near", metadata={"corpus_version_id": marker + "-other", "page": 3}),
    ]
    try:
        assert store.add_documents(documents, ids=ids) == ids
        corpus_filter = FieldFilter("metadata.corpus_version_id", "==", marker)
        results = store.similarity_search_with_score("near", k=5, filters=corpus_filter)
        scored = {
            doc.metadata["reference"]["path"].rsplit("/", 1)[-1]: (doc, score)
            for doc, score in results
        }
        assert set(scored) == set(ids[:2]), scored
        assert all(math.isfinite(score) for _, score in scored.values())
        assert math.isclose(scored[ids[0]][1], 1.0, abs_tol=0.02)
        assert math.isclose(scored[ids[1]][1], 0.0, abs_tol=0.02)
        assert scored[ids[0]][0].metadata["metadata"]["page"] == 1

        direct = collection.where(filter=corpus_filter).find_nearest(
            vector_field="embedding",
            query_vector=Vector(DeterministicEmbeddings._vector("near")),
            distance_measure=DistanceMeasure.COSINE,
            limit=5,
            distance_result_field="direct_distance",
        ).get()
        direct_scores = {
            snapshot.id: 1.0 - float(snapshot.to_dict()["direct_distance"])
            for snapshot in direct
        }
        assert set(direct_scores) == set(scored)
        for doc_id, (_, score) in scored.items():
            assert math.isclose(score, direct_scores[doc_id], abs_tol=1e-6)
        print("Live Firestore filter, metadata, cosine distance and native scores: passed")
    finally:
        for doc_id in ids:
            collection.document(doc_id).delete()


if __name__ == "__main__":
    main()

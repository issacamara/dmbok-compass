import base64
import math
import openrouter
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.vector import Vector
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.01] * 768 for _ in texts]

    def embed_query(self, text):
        return [0.01] * 768


class FakeDocumentRef:
    def __init__(self, doc_id):
        self.id = doc_id
        self.path = f"document_chunks_v2/{doc_id}"


class FakeSnapshot:
    def __init__(self, doc_id, data):
        self.reference = FakeDocumentRef(doc_id)
        self.data = data

    def to_dict(self):
        return dict(self.data)


class FakeBatch:
    def __init__(self, client):
        self.client = client
        self.pending = []

    def set(self, doc, data, merge=False):
        self.pending.append((doc.id, data, merge))

    def commit(self):
        self.client.commits.append(self.pending)


class FakeCollection:
    def __init__(self):
        self.docs = {}
        self.last_filter = None
        self.last_nearest = None

    def document(self, doc_id=None):
        return FakeDocumentRef(doc_id)

    def where(self, filter):
        self.last_filter = filter
        return self

    def find_nearest(self, **kwargs):
        self.last_nearest = kwargs
        doc_id = "chunk-500"
        data = dict(self.docs[doc_id])
        data[kwargs["distance_result_field"]] = 0.2
        return SimpleNamespace(get=lambda: [FakeSnapshot(doc_id, data)])


class FakeClient:
    def __init__(self):
        self._client_info = SimpleNamespace(user_agent="")
        self.commits = []

    def batch(self):
        return FakeBatch(self)


def test_pdf_and_splitter():
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/corpus/searchable-provenance-fixture.pdf.b64"
    with TemporaryDirectory() as temp_dir:
        pdf = Path(temp_dir) / "source.pdf"
        pdf.write_bytes(base64.b64decode(fixture.read_text()))
        pages = PyPDFLoader(str(pdf), mode="page").load()
    assert len(pages) == 2
    assert [p.metadata["page"] for p in pages] == [0, 1]
    assert "Purpose" in pages[0].page_content
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=700,
        chunk_overlap=70,
        length_function=lambda text: len(text.split()),
    )
    synthetic = Document(page_content=" ".join(f"w{i}" for i in range(1400)), metadata={"page": 0})
    chunks = splitter.split_documents([synthetic])
    lengths = [len(chunk.page_content.split()) for chunk in chunks]
    assert max(lengths) <= 700, lengths
    assert all(chunk.metadata["page"] == 0 for chunk in chunks)
    assert chunks[0].page_content.split()[-70:] == chunks[1].page_content.split()[:70]
    print("PDF page mode and word-sized overlap:", lengths)


def test_embeddings_configuration():
    calls = []

    def embed_content(**kwargs):
        calls.append(kwargs)
        contents = kwargs["contents"]
        count = len(contents) if isinstance(contents, list) else 1
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.01] * 768) for _ in range(count)])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        vertexai=True,
        project="spike-project",
        location="us-central1",
        credentials=AnonymousCredentials(),
        output_dimensionality=768,
    )
    assert embeddings._use_vertexai is True
    embeddings.client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    assert len(embeddings.embed_documents(["document"])) == 1
    assert len(embeddings.embed_query("query")) == 768
    assert [call["config"].task_type for call in calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert all(call["config"].output_dimensionality == 768 for call in calls)
    print("Vertex document/query embedding config and 768-dimensional responses: passed")


def test_store():
    client = FakeClient()
    collection = FakeCollection()
    store = FirestoreVectorStore(collection, FakeEmbeddings(), client=client, content_field="text")
    documents = [
        Document(page_content=f"text {i}", metadata={"corpus_version_id": "v2", "page": 1})
        for i in range(501)
    ]
    ids = [f"chunk-{i}" for i in range(501)]
    returned = store.add_documents(documents, ids=ids)
    assert returned == ids
    assert [len(batch) for batch in client.commits] == [500, 1]
    for batch in client.commits:
        for doc_id, data, merge in batch:
            assert merge
            collection.docs[doc_id] = data
    assert collection.docs["chunk-500"]["text"] == "text 500"
    assert collection.docs["chunk-500"]["metadata"]["corpus_version_id"] == "v2"
    assert isinstance(collection.docs["chunk-500"]["embedding"], Vector)
    corpus_filter = FieldFilter("metadata.corpus_version_id", "==", "v2")
    results = store.similarity_search_with_score("question", k=5, filters=corpus_filter)
    assert len(results) == 1
    doc, score = results[0]
    assert math.isclose(score, 0.8)
    assert doc.page_content == "text 500"
    assert doc.metadata["metadata"] == {"corpus_version_id": "v2", "page": 1}
    assert doc.metadata["reference"]["path"] == "document_chunks_v2/chunk-500"
    assert "langchain_distance" not in doc.metadata
    assert collection.last_filter is corpus_filter
    assert collection.last_nearest["limit"] == 5
    assert collection.last_nearest["distance_result_field"] == "langchain_distance"
    print("Native 501-document writes, IDs, nested provenance, corpus filter and score: passed")


def test_generation_construction():
    class Answer(BaseModel):
        answer: str

    sdk = openrouter.OpenRouter(api_key="spike-test-key", retry_config=None, timeout_ms=8000)
    openrouter_model = ChatOpenRouter(
        model="openai/gpt-4o-mini", api_key="spike-test-key",
        client=sdk, max_retries=0, timeout=8000,
    )
    assert openrouter_model.client.sdk_configuration.retry_config is None
    assert openrouter_model.request_timeout == 8000
    openrouter_chain = openrouter_model.with_structured_output(Answer, method="json_schema")
    assert isinstance(openrouter_chain, Runnable)
    json_object_chain = (
        openrouter_model.bind(response_format={"type": "json_object"})
        | PydanticOutputParser(pydantic_object=Answer)
    )
    assert isinstance(json_object_chain, Runnable)
    vertex = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        vertexai=True,
        project="spike-project",
        location="us-central1",
        credentials=AnonymousCredentials(),
        retries=0,
    )
    vertex_chain = vertex.with_structured_output(Answer)
    assert isinstance(vertex_chain, Runnable)
    print("OpenRouter and Vertex chat structured-output construction: passed")


def test_eligible_fallback_only():
    calls = []

    def primary(value):
        calls.append("primary")
        raise TimeoutError("eligible") if value == "timeout" else ValueError("ineligible")

    def fallback(value):
        calls.append("fallback")
        return "recovered"

    chain = RunnableLambda(primary).with_fallbacks(
        [RunnableLambda(fallback)], exceptions_to_handle=(TimeoutError,)
    )
    assert chain.invoke("timeout") == "recovered"
    assert calls == ["primary", "fallback"]
    calls.clear()
    try:
        chain.invoke("invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("ineligible error unexpectedly triggered fallback")
    assert calls == ["primary"]
    print("LangChain fallback composition honors eligible exceptions: passed")


test_pdf_and_splitter()
test_embeddings_configuration()
test_store()
test_generation_construction()
test_eligible_fallback_only()

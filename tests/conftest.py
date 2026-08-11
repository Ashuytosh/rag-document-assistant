"""Shared fixtures.

Fixture documents are generated rather than committed as binaries, so the suite stays
dependency-light and deterministic. Extraction runs locally, so nothing is mocked.
"""

import hashlib
import io
import struct
from collections.abc import Iterator
from pathlib import Path

import docx
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app import main
from app.config import Settings, get_settings
from app.main import app
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService

PDF_SAMPLE_TEXT = "Retrieval augmented generation test document."
DOCX_PARAGRAPHS = ("Grounded answers require citations.", "Every passage carries a source.")

#: Small enough that the oversize test is instant, large enough that the valid
#: fixtures still fit comfortably — a minimal DOCX is a zip container of ~36 KB.
TEST_MAX_UPLOAD_BYTES = 128 * 1024


def _build_minimal_pdf(text: str) -> bytes:
    """Assemble a single-page PDF containing ``text`` in Helvetica.

    Hand-built so the suite needs no PDF authoring library; pypdf extracts the text
    back out verbatim.
    """
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")

    xref_offset = buffer.tell()
    size = len(objects) + 1
    buffer.write(f"xref\n0 {size}\n".encode())
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets:
        buffer.write(f"{offset:010d} 00000 n \n".encode())
    buffer.write(
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return buffer.getvalue()


def _build_docx(paragraphs: tuple[str, ...]) -> bytes:
    """Assemble a DOCX containing one paragraph per entry."""
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    return _build_minimal_pdf(PDF_SAMPLE_TEXT)


@pytest.fixture(scope="session")
def sample_docx_bytes() -> bytes:
    return _build_docx(DOCX_PARAGRAPHS)


@pytest.fixture(scope="session")
def corrupt_pdf_bytes() -> bytes:
    """A PDF header followed by garbage — well-formed enough to be accepted, not to parse."""
    return b"%PDF-1.4\n" + b"\x00\x01\x02not a real pdf body\x03\x04" * 8


@pytest.fixture(scope="session")
def empty_pdf_bytes() -> bytes:
    """A structurally valid single-page PDF whose only text run is empty."""
    return _build_minimal_pdf("")


@pytest.fixture(scope="session")
def empty_docx_bytes() -> bytes:
    """A structurally valid DOCX containing only blank paragraphs."""
    return _build_docx(("", "   ", "\t"))


@pytest.fixture(scope="session")
def long_docx_bytes() -> bytes:
    """A DOCX whose text comfortably exceeds the preview length."""
    return _build_docx(("word " * 400,))


class FakeEmbeddings(Embeddings):
    """Deterministic, dependency-free stand-in for the sentence-transformers model.

    Vectors are derived from a hash of the text and unit-normalized, so cosine distance
    behaves the way it does with the real model: identical text scores 1.0, unrelated
    text scores lower. Word overlap is blended in so that a query sharing words with a
    chunk lands nearer to it than to an unrelated chunk — enough to make ranking
    assertions meaningful without loading torch.
    """

    dimensions = 16

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in text.lower().split() or [text.lower()]:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            for index in range(self.dimensions):
                (component,) = struct.unpack_from("<h", digest, index * 2)
                vector[index] += component / 32768.0
        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude == 0:
            return [1.0] + [0.0] * (self.dimensions - 1)
        return [value / magnitude for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    """An isolated Chroma directory, so tests never touch the real chroma_db/."""
    return tmp_path / "chroma"


@pytest.fixture
def embedding_service(test_settings: Settings, fake_embeddings: FakeEmbeddings) -> EmbeddingService:
    """The real service class, wrapping a fake model — no torch, no download."""
    return EmbeddingService(test_settings, embeddings=fake_embeddings)


@pytest.fixture
def vector_store(
    test_settings: Settings, embedding_service: EmbeddingService
) -> VectorStoreService:
    """A real Chroma collection in a temp directory, backed by the fake embeddings."""
    return VectorStoreService(test_settings, embedding_service.as_langchain())


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    """An isolated upload directory, so tests never touch the real data/uploads/."""
    target = tmp_path / "uploads"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def test_settings(upload_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        upload_dir=upload_dir,
        max_upload_bytes=TEST_MAX_UPLOAD_BYTES,
        chroma_persist_dir=tmp_path / "chroma",
    )


@pytest.fixture
def client(
    test_settings: Settings,
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """A TestClient wired to per-test storage and a fake embedding model.

    The dependency override covers request handling; the ``app.main`` patches cover the
    lifespan, which resolves settings and builds the services directly. Without the
    factory patches every test would load torch and download the real model.
    """
    monkeypatch.setattr(main, "get_settings", lambda: test_settings)
    monkeypatch.setattr(main, "build_embedding_service", lambda _settings: embedding_service)
    monkeypatch.setattr(main, "build_vector_store", lambda _settings, _embeddings: vector_store)
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

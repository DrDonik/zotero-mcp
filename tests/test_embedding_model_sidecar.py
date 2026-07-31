"""Fork feature: provider-aware embedding-model tracking via a sidecar file.

ChromaDB's own persisted collection config only records a ``model_name``, and
only for backends that expose one. That is not enough to notice a provider
switch — ``openai:text-embedding-3-small`` and a local model of the same name
look identical, and the default MiniLM backend records nothing at all.

``ChromaClient`` therefore also writes the resolved ``provider:model`` string to
``.embedding_model_info`` in the persist directory and rebuilds the collection
when that value changes, because embeddings from different models are not
comparable and usually differ in dimension.
"""

import json

import pytest

chromadb = pytest.importorskip("chromadb")  # noqa: F841

from zotero_mcp.chroma_client import ChromaClient  # noqa: E402


def _client(tmp_path, embedding_model="default", embedding_config=None):
    return ChromaClient(
        collection_name="zotero_library",
        persist_directory=str(tmp_path),
        embedding_model=embedding_model,
        embedding_config=embedding_config,
    )


def test_sidecar_is_written_on_first_use(tmp_path):
    client = _client(tmp_path)

    assert client.metadata_file.exists()
    data = json.loads(client.metadata_file.read_text())
    assert data["embedding_model_info"] == "default"


def test_reopening_with_the_same_model_keeps_the_collection(tmp_path):
    first = _client(tmp_path)
    first.collection.add(ids=["ITEM001"], documents=["hello world"])
    assert first.collection.count() == 1

    second = _client(tmp_path)

    assert second.collection.count() == 1, (
        "an unchanged embedding model must not trigger a rebuild"
    )


def test_switching_provider_rebuilds_the_collection(tmp_path, capsys, monkeypatch):
    # Keep the embedding function fixed so only the *label* changes: that is
    # exactly the case ChromaDB's persisted config cannot detect and the sidecar
    # must. It also keeps the test offline — no model weights are fetched.
    shared_ef = ChromaClient.__new__(ChromaClient)
    shared_ef.embedding_model = "default"
    shared_ef.embedding_config = {}
    shared_ef = ChromaClient._create_embedding_function(shared_ef)
    monkeypatch.setattr(
        ChromaClient, "_create_embedding_function", lambda self: shared_ef
    )

    first = _client(tmp_path)
    first.collection.add(ids=["ITEM001"], documents=["hello world"])
    assert first.collection.count() == 1

    second = _client(
        tmp_path,
        embedding_model="openai",
        embedding_config={"model_name": "text-embedding-3-small"},
    )

    assert second.collection.count() == 0, (
        "a changed embedding model must drop the stale vectors"
    )
    assert "mismatch" in capsys.readouterr().err.lower()
    data = json.loads(second.metadata_file.read_text())
    assert data["embedding_model_info"] == "openai:text-embedding-3-small"


@pytest.mark.parametrize(
    "embedding_model, embedding_config, expected",
    [
        ("default", None, "default"),
        ("openai", None, "openai:text-embedding-3-small"),
        ("openai", {"model_name": "text-embedding-3-large"}, "openai:text-embedding-3-large"),
        ("gemini", None, "gemini:gemini-embedding-001"),
        ("ollama", None, "ollama:qwen3-embedding"),
        ("qwen", None, "qwen:Qwen/Qwen3-Embedding-0.6B"),
        ("embeddinggemma", None, "embeddinggemma:google/embeddinggemma-300m"),
        ("some/hf-model", None, "huggingface:some/hf-model"),
    ],
)
def test_model_info_defaults_match_create_embedding_function(
    embedding_model, embedding_config, expected
):
    """A config that omits model_name must not look like a change next run.

    ``_get_current_model_info`` hardcodes the same fallbacks as
    ``_create_embedding_function``; if the two drift apart, every start-up
    reports a spurious mismatch and wipes the index.
    """
    stub = ChromaClient.__new__(ChromaClient)
    stub.embedding_model = embedding_model
    stub.embedding_config = embedding_config or {}

    assert stub._get_current_model_info() == expected

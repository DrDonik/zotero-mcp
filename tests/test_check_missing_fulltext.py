"""Tests for the `check-missing-fulltext` report.

The report predates chunking, when one item meant one document. With
chunking on, an item's text is spread over many passages that all carry the
parent's `item_key`, so a per-document view sees only the last passage —
roughly 1500 characters, comfortably above the "too short" threshold — and
reports a truncated item as healthy. These pin the per-item aggregation that
replaced it, plus the `has_fulltext="failed"` marker the indexer writes when
extraction was tried and gave up.
"""

from zotero_mcp.check_missing_fulltext import (
    aggregate_indexed_items,
    classify_item,
    find_missing_fulltext,
)


class FakeCollection:
    """Minimal stand-in for a ChromaDB collection's `get()` contract."""

    def __init__(self, records):
        # records: list of (id, document, metadata)
        self._records = list(records)

    def get(self, ids=None, include=None):
        include = include or []
        if ids is None:
            selected = self._records
        else:
            wanted = set(ids)
            selected = [r for r in self._records if r[0] in wanted]
        result = {"ids": [r[0] for r in selected]}
        if "documents" in include:
            result["documents"] = [r[1] for r in selected]
        if "metadatas" in include:
            result["metadatas"] = [r[2] for r in selected]
        return result


def _chunks(item_key, n, chunk_chars=1500, **meta):
    return [
        (f"{item_key}#{i}", "x" * chunk_chars,
         {"item_key": item_key, "parent_item_key": item_key, "chunk_index": i,
          "n_chunks": n, "has_fulltext": True, **meta})
        for i in range(n)
    ]


def test_chunks_of_one_item_sum_into_a_single_record():
    collection = FakeCollection(_chunks("AAAA1111", 40))

    indexed = aggregate_indexed_items(collection, batch_size=7)

    assert set(indexed) == {"AAAA1111"}
    assert indexed["AAAA1111"]["chunks"] == 40
    assert indexed["AAAA1111"]["length"] == 40 * 1500


def test_paging_covers_every_document():
    """A batch size that does not divide the id count must not drop the tail."""
    collection = FakeCollection(_chunks("AAAA1111", 10) + _chunks("BBBB2222", 3))

    indexed = aggregate_indexed_items(collection, batch_size=4)

    assert indexed["AAAA1111"]["chunks"] == 10
    assert indexed["BBBB2222"]["chunks"] == 3


def test_a_healthy_chunked_item_is_not_reported():
    collection = FakeCollection(_chunks("AAAA1111", 40))
    pdf_items = {"AAAA1111": {"item_key": "AAAA1111", "title": "T", "author": "A", "date": "D"}}

    failures = find_missing_fulltext(pdf_items, aggregate_indexed_items(collection))

    assert failures == []


def test_failed_marker_wins_over_a_long_document():
    """`has_fulltext="failed"` is a string, so a truthiness test reads it as
    success — exactly backwards. One marked passage condemns the item."""
    records = _chunks("AAAA1111", 5)
    records[2][2]["has_fulltext"] = "failed"
    collection = FakeCollection(records)

    indexed = aggregate_indexed_items(collection)

    assert indexed["AAAA1111"]["has_fulltext"] == "failed"
    reason, length, chunks = classify_item(indexed["AAAA1111"])
    assert reason == "Extraction previously failed (marked in index)"
    assert (length, chunks) == (5 * 1500, 5)


def test_short_and_empty_and_absent_items_are_reported():
    collection = FakeCollection([
        ("SHORT111#0", "too short", {"item_key": "SHORT111", "has_fulltext": True}),
        ("EMPTY222#0", "   ", {"item_key": "EMPTY222", "has_fulltext": True}),
        ("NOMARK33#0", "y" * 4000, {"item_key": "NOMARK33"}),
    ])
    pdf_items = {
        k: {"item_key": k, "title": "T", "author": "A", "date": "D"}
        for k in ("SHORT111", "EMPTY222", "NOMARK33", "ABSENT44")
    }

    failures = find_missing_fulltext(pdf_items, aggregate_indexed_items(collection))
    reasons = {f["item_key"]: f["reason"] for f in failures}

    assert reasons == {
        "SHORT111": "Very short fulltext (< 100 chars)",
        "EMPTY222": "Empty fulltext in ChromaDB",
        "NOMARK33": "Metadata indicates no fulltext",
        "ABSENT44": "Not indexed in ChromaDB",
    }


def test_item_key_falls_back_to_the_id_prefix():
    """Documents written before the metadata carried item_key still group."""
    collection = FakeCollection([
        ("LEGACY11#0", "z" * 2000, {"has_fulltext": True}),
        ("LEGACY11#1", "z" * 2000, {"has_fulltext": True}),
    ])

    indexed = aggregate_indexed_items(collection)

    assert indexed["LEGACY11"]["chunks"] == 2


def test_zotero_db_is_opened_immutable(monkeypatch):
    """Zotero holds a write lock while it runs, and `mode=ro` is not exempt:
    the report died with "database is locked" whenever the app was open."""
    import sqlite3

    from zotero_mcp import check_missing_fulltext as cmf

    seen = {}
    real_connect = sqlite3.connect

    def fake_connect(dsn, *args, **kwargs):
        seen["dsn"] = dsn
        return real_connect(":memory:")

    monkeypatch.setattr(cmf, "find_zotero_db", lambda: "/tmp/zotero.sqlite")
    monkeypatch.setattr(sqlite3, "connect", fake_connect)

    cmf.open_zotero_db()

    assert "immutable=1" in seen["dsn"]
    assert "mode=ro" not in seen["dsn"]


def _fixture_zotero_db(path, trashed_keys=()):
    """A minimal zotero.sqlite holding two PDF-owning items."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INTEGER);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, orderIndex INTEGER);
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        CREATE TABLE itemAttachments (itemID INTEGER, parentItemID INTEGER, contentType TEXT);
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        INSERT INTO itemTypes VALUES (1, 'journalArticle'), (2, 'attachment');
    """)
    for i, key in enumerate(("KEEPME01", "TRASHED1"), start=1):
        conn.execute("INSERT INTO items VALUES (?,?,1)", (i, key))
        conn.execute("INSERT INTO itemDataValues VALUES (?,?)", (i, f"Title {key}"))
        conn.execute("INSERT INTO itemData VALUES (?,1,?)", (i, i))
        conn.execute("INSERT INTO items VALUES (?,?,2)", (100 + i, f"ATT{i}"))
        conn.execute("INSERT INTO itemAttachments VALUES (?,?,'application/pdf')", (100 + i, i))
        if key in trashed_keys:
            conn.execute("INSERT INTO deletedItems VALUES (?)", (i,))
    conn.commit()
    conn.close()


def test_trashed_items_are_not_reported_as_missing(tmp_path, monkeypatch):
    """Items in Zotero's trash are excluded from indexing by design, so listing
    them as "not indexed" reports the user's own deletion back as a failure."""
    from zotero_mcp import check_missing_fulltext as cmf

    db = tmp_path / "zotero.sqlite"
    _fixture_zotero_db(db, trashed_keys=("TRASHED1",))
    monkeypatch.setattr(cmf, "find_zotero_db", lambda: str(db))

    items = cmf.get_items_with_pdf_attachments()

    assert set(items) == {"KEEPME01"}


def _article(item_key, chars=4000, **meta):
    return (item_key, "a" * chars, {"item_key": item_key, "has_fulltext": True, **meta})


def test_split_index_counts_passages_from_their_own_collection():
    """0.17.0 keeps one article record per item and the passages apart."""
    articles = FakeCollection([_article("AAAA1111"), _article("BBBB2222")])
    passages = FakeCollection(_chunks("AAAA1111", 5) + _chunks("BBBB2222", 2))

    indexed = aggregate_indexed_items([articles, passages], batch_size=3)

    # The article record is a truncated copy of the same text: not added on top.
    assert indexed["AAAA1111"]["chunks"] == 5
    assert indexed["AAAA1111"]["length"] == 5 * 1500
    assert indexed["BBBB2222"]["chunks"] == 2
    assert indexed["BBBB2222"]["length"] == 2 * 1500


def test_article_record_alone_counts_as_one_document():
    """Chunking off, or passages not filled in yet: the article record speaks."""
    indexed = aggregate_indexed_items([FakeCollection([_article("CCCC3333", 2500)])])

    assert indexed["CCCC3333"] == {"length": 2500, "chunks": 1, "has_fulltext": True}


def test_failed_marker_on_the_article_record_is_reported_in_a_split_index():
    articles = FakeCollection([_article("DDDD4444", 300, has_fulltext="failed")])
    passages = FakeCollection([])
    pdf_items = {"DDDD4444": {"item_key": "DDDD4444", "title": "T", "author": "A", "date": "D"}}

    failures = find_missing_fulltext(pdf_items, aggregate_indexed_items([articles, passages]))

    assert [f["reason"] for f in failures] == ["Extraction previously failed (marked in index)"]


def test_unmigrated_chunked_index_is_read_like_before():
    """Passages still in the article collection, next to article records."""
    mixed = FakeCollection([_article("EEEE5555")] + _chunks("EEEE5555", 3))

    indexed = aggregate_indexed_items(mixed)

    assert indexed["EEEE5555"]["chunks"] == 3
    assert indexed["EEEE5555"]["length"] == 3 * 1500


def test_report_reads_every_collection_of_the_index(monkeypatch, capsys):
    import zotero_mcp.check_missing_fulltext as cmf
    import zotero_mcp.chroma_client as cc

    articles = FakeCollection([_article("FFFF6666", chars=50)])
    passages = FakeCollection(_chunks("FFFF6666", 4))

    class FakeClient:
        collection = articles

        def collections(self):
            return [articles, passages]

    monkeypatch.setattr(cmf, "get_items_with_pdf_attachments", lambda: {
        "FFFF6666": {"item_key": "FFFF6666", "title": "T", "author": "A", "date": "D"},
    })
    monkeypatch.setattr(cc, "create_chroma_client", lambda config_path=None: FakeClient())

    cmf.check_chromadb_fulltext(output_format="json")

    assert "Found 0 items with extraction failures" in capsys.readouterr().err


def test_real_split_index_is_aggregated_across_both_collections(tmp_path):
    """On a real persistent index with a separate passage collection (0.17.0)."""
    import importlib.util
    import sys

    import pytest

    if sys.version_info >= (3, 14) or importlib.util.find_spec("chromadb") is None:
        pytest.skip("needs chromadb on Python < 3.14")

    from zotero_mcp.chroma_client import ChromaClient

    client = ChromaClient(collection_name="cmf_test", persist_directory=str(tmp_path / "chroma"))
    client.route_passages = True
    ids = ["GGGG7777", "GGGG7777#0", "GGGG7777#1", "HHHH8888"]
    docs = ["article " * 50, "p" * 1200, "q" * 800, ""]
    metas = [
        {"item_key": "GGGG7777", "has_fulltext": True},
        {"item_key": "GGGG7777", "parent_item_key": "GGGG7777", "chunk_index": 0,
         "n_chunks": 2, "has_fulltext": True},
        {"item_key": "GGGG7777", "parent_item_key": "GGGG7777", "chunk_index": 1,
         "n_chunks": 2, "has_fulltext": True},
        {"item_key": "HHHH8888", "has_fulltext": "failed"},
    ]
    client.upsert_embeddings(documents=docs, metadatas=metas, ids=ids,
                             embeddings=[[1.0, 0.0] for _ in ids])

    collections = client.collections()
    assert len(collections) == 2  # the passages really went to their own collection

    indexed = aggregate_indexed_items(collections)

    assert indexed["GGGG7777"] == {"length": 2000, "chunks": 2, "has_fulltext": True}
    assert indexed["HHHH8888"]["has_fulltext"] == "failed"

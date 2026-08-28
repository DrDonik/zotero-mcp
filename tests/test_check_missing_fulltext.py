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

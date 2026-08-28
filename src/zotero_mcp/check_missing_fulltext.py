#!/usr/bin/env python3
"""Report items whose PDFs did not make it into the semantic index.

Answers one question: which items own a PDF attachment but carry no usable
full text in ChromaDB? It reads the existing index rather than re-extracting,
so it is cheap to run after an `update-db --fulltext` to see what the run
could not reach.

Chunking (`semantic_search.chunking`) stores one document per passage, all
sharing the parent's `item_key`, so every figure here is aggregated per item:
an item's length is the sum over its passages, and a single passage's
`has_fulltext` marker speaks for the whole item.
"""

import csv
import json
import sqlite3
import sys
from typing import Any, Iterator

from .local_db import LocalZoteroReader

# Below this many characters an item is treated as an extraction failure
# rather than a short document: a real PDF that indexed correctly clears it
# by two orders of magnitude.
_MIN_USABLE_CHARS = 100


def find_zotero_db():
    """Find the Zotero database location.

    Delegates to LocalZoteroReader so this command honours ZOTERO_DB_PATH and a
    custom dataDir from Zotero's preferences, exactly like semantic search does.
    """
    return LocalZoteroReader().db_path


def open_zotero_db():
    """A read-only connection that works while Zotero itself is running.

    Zotero keeps its database in rollback-journal mode and holds a write lock
    for as long as the app is open, which blocks even a `mode=ro` connection
    with "database is locked". `immutable=1` skips lock checking altogether —
    the same trade `LocalZoteroReader._get_connection` makes, and for the same
    reason: this path only ever reads, and tolerating slightly stale data
    beats refusing to run unless the user quits Zotero.
    """
    conn = sqlite3.connect(f"file:{find_zotero_db()}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_items_with_pdf_attachments():
    """Get all Zotero items that have PDF attachments."""
    conn = open_zotero_db()

    # Get items with PDF attachments
    query = """
    SELECT DISTINCT
        i.itemID,
        i.key,
        title_val.value as title,
        date_val.value as date,
        creator.lastName || ', ' || creator.firstName as author
    FROM items i
    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
    LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
    LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID
    LEFT JOIN itemData date_data ON i.itemID = date_data.itemID AND date_data.fieldID = 14
    LEFT JOIN itemDataValues date_val ON date_data.valueID = date_val.valueID
    LEFT JOIN itemCreators ic ON i.itemID = ic.itemID AND ic.orderIndex = 0
    LEFT JOIN creators creator ON ic.creatorID = creator.creatorID
    WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
        AND EXISTS (
            SELECT 1 FROM itemAttachments ia
            WHERE ia.parentItemID = i.itemID
            AND ia.contentType = 'application/pdf'
        )
    """

    cursor = conn.execute(query)
    items = cursor.fetchall()

    result = {}
    for item in items:
        result[item['key']] = {
            'item_key': item['key'],
            'title': item['title'] or 'Untitled',
            'author': item['author'] or 'No author',
            'date': item['date'] or 'No date'
        }

    conn.close()
    return result


def iter_indexed_documents(collection, batch_size: int = 500) -> Iterator[tuple[list[str], dict]]:
    """Stream ``(ids, result)`` over the whole collection in bounded batches.

    Snapshots the id set first and pages it with ``get(ids=chunk)``, for the
    same reason ``ChromaClient.iter_metadatas`` does: one unbounded ``get()``
    binds a SQL variable per row and trips SQLite's ceiling on a large index —
    and on a chunked collection it would also pull every passage's text into
    memory at once.
    """
    batch_size = max(1, min(int(batch_size), 5000))
    all_ids = sorted(collection.get(include=[]).get("ids") or [])
    for start in range(0, len(all_ids), batch_size):
        chunk = all_ids[start:start + batch_size]
        yield chunk, collection.get(ids=chunk, include=["documents", "metadatas"])


def aggregate_indexed_items(collection, batch_size: int = 500) -> dict[str, dict[str, Any]]:
    """Fold the collection into one record per Zotero item.

    Returns ``{item_key: {"length", "chunks", "has_fulltext"}}`` where length
    sums the item's passages. ``has_fulltext`` keeps the strongest marker any
    passage carries: ``"failed"`` (extraction was tried and gave up) outranks
    ``True``, which outranks a missing marker.
    """
    items: dict[str, dict[str, Any]] = {}
    for _chunk_ids, result in iter_indexed_documents(collection, batch_size):
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            metadata = metadata or {}
            # Chunks inherit the parent's item_key; parent_item_key and the
            # `KEY#n` id shape are the fallbacks if that ever stops holding.
            item_key = (
                metadata.get("item_key")
                or metadata.get("parent_item_key")
                or str(doc_id).split("#", 1)[0]
            )
            if not item_key:
                continue
            doc = documents[i] if i < len(documents) else ""
            record = items.setdefault(
                item_key, {"length": 0, "chunks": 0, "has_fulltext": False}
            )
            record["length"] += len((doc or "").strip())
            record["chunks"] += 1
            marker = metadata.get("has_fulltext", False)
            if record["has_fulltext"] != "failed":
                if marker == "failed":
                    record["has_fulltext"] = "failed"
                elif marker:
                    record["has_fulltext"] = True
    return items


def classify_item(indexed: dict[str, Any] | None) -> tuple[str, int, int] | None:
    """Return ``(reason, length, chunks)`` if this item failed, else None."""
    if indexed is None:
        return ("Not indexed in ChromaDB", 0, 0)

    length = indexed["length"]
    chunks = indexed["chunks"]

    if indexed["has_fulltext"] == "failed":
        return ("Extraction previously failed (marked in index)", length, chunks)
    if length == 0:
        return ("Empty fulltext in ChromaDB", length, chunks)
    if length < _MIN_USABLE_CHARS:
        return (f"Very short fulltext (< {_MIN_USABLE_CHARS} chars)", length, chunks)
    if not indexed["has_fulltext"]:
        return ("Metadata indicates no fulltext", length, chunks)
    return None


def find_missing_fulltext(pdf_items: dict[str, dict], indexed: dict[str, dict[str, Any]]) -> list[dict]:
    """Join the Zotero-side PDF owners against the index and keep the failures."""
    failures = []
    for item_key, item_info in pdf_items.items():
        verdict = classify_item(indexed.get(item_key))
        if verdict is None:
            continue
        reason, length, chunks = verdict
        failures.append({
            **item_info,
            'reason': reason,
            'fulltext_length': length,
            'chunks': chunks,
        })
    return failures


def check_chromadb_fulltext(output_format="text", config_path=None):
    """Check ChromaDB for items with PDFs but missing/empty fulltext."""

    # Get items with PDF attachments from Zotero DB
    print("Loading items with PDF attachments from Zotero database...", file=sys.stderr)
    pdf_items = get_items_with_pdf_attachments()
    print(f"Found {len(pdf_items)} items with PDF attachments", file=sys.stderr)

    # Load ChromaDB. Imported here, not at module scope, so the report's
    # aggregation logic stays importable (and testable) without chromadb —
    # the same reason the server defers its own ChromaDB import (#485).
    from .chroma_client import create_chroma_client

    print("Loading ChromaDB...", file=sys.stderr)
    chroma_client = create_chroma_client(config_path)

    print("Querying ChromaDB for fulltext data...", file=sys.stderr)
    indexed = aggregate_indexed_items(chroma_client.collection)
    print(f"Found {len(indexed)} items in ChromaDB", file=sys.stderr)

    failures = find_missing_fulltext(pdf_items, indexed)
    print(f"\nFound {len(failures)} items with extraction failures\n", file=sys.stderr)

    # Output in requested format
    if output_format == "csv":
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=['item_key', 'title', 'author', 'date', 'reason', 'fulltext_length', 'chunks']
        )
        writer.writeheader()
        writer.writerows(failures)
    elif output_format == "json":
        print(json.dumps(failures, indent=2))
    else:  # text
        if failures:
            print(f"Found {len(failures)} items with PDF attachments but missing/incomplete fulltext in ChromaDB")
            print()
            for f in failures:
                print(f"Item Key: {f['item_key']}")
                print(f"Title: {f['title']}")
                print(f"Author: {f['author']}")
                print(f"Date: {f['date']}")
                print(f"Reason: {f['reason']}")
                print(f"Fulltext Length: {f['fulltext_length']} characters in {f['chunks']} passage(s)")
                print("-" * 70)
        else:
            print("No items found with missing fulltext!")

    return failures


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Check ChromaDB for PDFs with missing/empty fulltext"
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--config-path",
        help="Path to semantic search configuration file"
    )
    args = parser.parse_args()

    check_chromadb_fulltext(output_format=args.format, config_path=args.config_path)


if __name__ == "__main__":
    main()

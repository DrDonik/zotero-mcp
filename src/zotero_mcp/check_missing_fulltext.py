#!/usr/bin/env python3
"""
Check ChromaDB for items with PDF attachments but missing/empty fulltext.
This queries the existing semantic search database instead of re-extracting.
"""

import csv
import json
import sqlite3
import sys

from .chroma_client import create_chroma_client
from .local_db import LocalZoteroReader


def find_zotero_db():
    """Find the Zotero database location.

    Delegates to LocalZoteroReader so this command honours ZOTERO_DB_PATH and a
    custom dataDir from Zotero's preferences, exactly like semantic search does.
    """
    return LocalZoteroReader().db_path


def get_items_with_pdf_attachments():
    """Get all Zotero items that have PDF attachments."""
    db_path = find_zotero_db()
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

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


def check_chromadb_fulltext(output_format="text"):
    """Check ChromaDB for items with PDFs but missing/empty fulltext."""

    # Get items with PDF attachments from Zotero DB
    print("Loading items with PDF attachments from Zotero database...", file=sys.stderr)
    pdf_items = get_items_with_pdf_attachments()
    print(f"Found {len(pdf_items)} items with PDF attachments", file=sys.stderr)

    # Load ChromaDB
    print("Loading ChromaDB...", file=sys.stderr)
    chroma_client = create_chroma_client()
    collection = chroma_client.collection

    # Get all documents from ChromaDB with their fulltext
    print("Querying ChromaDB for fulltext data...", file=sys.stderr)

    # Get all items from collection
    # ChromaDB doesn't support getting all at once easily, so we query with a large limit
    result = collection.get(
        include=["documents", "metadatas"]
    )

    # Build a map of item_key -> fulltext
    chromadb_data = {}
    for i, item_id in enumerate(result['ids']):
        doc = result['documents'][i] if i < len(result['documents']) else ""
        metadata = result['metadatas'][i] if i < len(result['metadatas']) else {}
        item_key = metadata.get('item_key', '')

        if item_key:
            chromadb_data[item_key] = {
                'document': doc,
                'has_fulltext': metadata.get('has_fulltext', False),
                'metadata': metadata
            }

    print(f"Found {len(chromadb_data)} items in ChromaDB", file=sys.stderr)

    # Find items with PDFs but missing/empty fulltext
    failures = []

    for item_key, item_info in pdf_items.items():
        if item_key not in chromadb_data:
            # Item not in ChromaDB at all
            failures.append({
                **item_info,
                'reason': 'Not indexed in ChromaDB',
                'fulltext_length': 0,
                'fulltext_preview': ''
            })
        else:
            chroma_item = chromadb_data[item_key]
            doc = chroma_item['document'] or ""
            doc_stripped = doc.strip()

            # Check if fulltext is missing or very short
            if not doc_stripped:
                failures.append({
                    **item_info,
                    'reason': 'Empty fulltext in ChromaDB',
                    'fulltext_length': 0,
                    'fulltext_preview': ''
                })
            elif len(doc_stripped) < 100:  # Very short text (likely extraction failed)
                failures.append({
                    **item_info,
                    'reason': 'Very short fulltext (< 100 chars)',
                    'fulltext_length': len(doc_stripped),
                    'fulltext_preview': doc_stripped[:100]
                })
            elif not chroma_item.get('has_fulltext', False):
                # Metadata indicates no fulltext
                failures.append({
                    **item_info,
                    'reason': 'Metadata indicates no fulltext',
                    'fulltext_length': len(doc_stripped),
                    'fulltext_preview': doc_stripped[:100]
                })

    print(f"\nFound {len(failures)} items with extraction failures\n", file=sys.stderr)

    # Output in requested format
    if output_format == "csv":
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=['item_key', 'title', 'author', 'date', 'reason', 'fulltext_length', 'fulltext_preview']
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
                print(f"Fulltext Length: {f['fulltext_length']} characters")
                if f['fulltext_preview']:
                    print(f"Preview: {f['fulltext_preview'][:100]}...")
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
    args = parser.parse_args()

    check_chromadb_fulltext(output_format=args.format)


if __name__ == "__main__":
    main()

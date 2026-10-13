"""
Bring news articles already in Pinecone up to the current storage format.

Articles stored before chunking was introduced sit in the index as one large
vector each, carrying no `timestamp`, which is what date filtering searches on.
This script finds them, re-fetches their pages, and stores them again as dated
passages under ids derived from the article URL.

Re-ingesting is safe on its own: the new passages are written under new ids and
the originals are left untouched, so nothing is lost if the run goes wrong. That
does mean both copies are searchable until the originals are removed, so once a
run looks right, repeat it with --delete-legacy.

    python -m scripts.reindex_news --dry-run
    python -m scripts.reindex_news
    python -m scripts.reindex_news --delete-legacy
"""

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(os.path.dirname(__file__)) / '..'))
from stock_assistant import get_pinecone_vector_store
from scripts.news_scraper import chunk_documents, process_urls, store_docs

load_dotenv()

index_name = os.getenv("INDEX_NAME")

# Pinecone rejects oversized fetch and delete calls, so work through them in pages.
BATCH_SIZE = 100


def read_vectors(index, ids):
    """Fetch a page of vectors, tolerating either client response shape."""
    fetched = index.fetch(ids=list(ids))
    return fetched.vectors if hasattr(fetched, "vectors") else fetched["vectors"]


def find_legacy_news(index):
    """
    Find news vectors that predate the current format.

    Returns:
        The articles to rebuild, keyed by URL, and the ids of the vectors they
        came from.
    """
    articles = {}
    legacy_ids = []

    for id_page in index.list():
        for vector_id, vector in read_vectors(index, id_page).items():
            metadata = getattr(vector, "metadata", None) or {}

            if metadata.get("source") != "news" or "timestamp" in metadata:
                continue

            legacy_ids.append(vector_id)

            link = metadata.get("link")
            if not link:
                print(f"Skipping {vector_id}: no link recorded, cannot re-fetch it.")
                continue
            if link in articles:
                continue

            articles[link] = {
                "title": metadata.get("title"),
                "link": link,
                "date": metadata.get("date"),
                "source": "news",
            }

    return articles, legacy_ids


def reingest(articles):
    """Re-fetch and store the given articles in the current format."""
    stored = 0
    for article in articles:
        # One article at a time: a page that has since gone missing should not
        # take the rest of the batch down with it.
        try:
            docs = process_urls([article])
            chunks, ids = chunk_documents(docs)
        except Exception as e:
            print(f"Could not rebuild {article['link']}: {e}")
            continue

        if not chunks:
            print(f"No readable text left at {article['link']}, skipping.")
            continue

        store_docs(chunks, ids)
        stored += len(chunks)

    return stored


def delete_legacy(index, legacy_ids):
    """Remove the pre-chunking vectors now that their replacements are stored."""
    for start in range(0, len(legacy_ids), BATCH_SIZE):
        batch = legacy_ids[start:start + BATCH_SIZE]
        try:
            index.delete(ids=batch)
        except Exception as e:
            print(f"Error deleting {len(batch)} old vectors: {e}")
            return
    print(f"Deleted {len(legacy_ids)} pre-chunking vectors.")


def main(dry_run: bool, should_delete: bool):
    index = get_pinecone_vector_store(index_name).index
    articles, legacy_ids = find_legacy_news(index)

    print(
        f"Found {len(legacy_ids)} pre-chunking news vectors "
        f"covering {len(articles)} articles."
    )
    if not legacy_ids:
        return

    if dry_run:
        for link in articles:
            print(f"  would rebuild {link}")
        if should_delete:
            print(f"  would then delete {len(legacy_ids)} old vectors")
        return

    stored = reingest(list(articles.values()))
    print(f"Stored {stored} passages from {len(articles)} articles.")

    if should_delete:
        if stored == 0:
            print("Nothing was re-ingested, so the old vectors are being kept.")
            return
        delete_legacy(index, legacy_ids)
    else:
        print(
            "Old vectors kept. Re-run with --delete-legacy once the results look "
            "right, otherwise both copies stay searchable."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-store legacy news articles as dated, chunked passages."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing anything",
    )
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="delete the pre-chunking vectors after their replacements are stored",
    )
    args = parser.parse_args()
    main(args.dry_run, args.delete_legacy)

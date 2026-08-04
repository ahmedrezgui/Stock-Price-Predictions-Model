from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from scripts.news_scraper import (
    CHUNK_SIZE,
    build_metadata,
    chunk_documents,
    chunk_id,
    main,
    to_iso_date,
    to_timestamp,
)

ARTICLE = {
    "title": "BIAT publishes its results",
    "link": "https://www.ilboursa.com/marches/article-1",
    "date": "01/03/2024",
    "source": "news",
}


def _document(content, article=None):
    return Document(page_content=content, metadata=build_metadata(article or ARTICLE))


# ------------------------------------------------------------------------- dates


def test_to_timestamp_is_utc_midnight():
    assert to_timestamp("01/03/2024") == 1709251200


def test_to_iso_date_reorders_the_parts():
    assert to_iso_date("01/03/2024") == "2024-03-01"


def test_dates_must_be_day_first():
    with pytest.raises(ValueError):
        to_timestamp("2024-03-01")


def test_build_metadata_keeps_the_readable_date_alongside_the_sortable_one():
    metadata = build_metadata(ARTICLE)

    assert metadata["date"] == "01/03/2024"
    assert metadata["date_iso"] == "2024-03-01"
    assert metadata["timestamp"] == 1709251200
    assert metadata["link"] == ARTICLE["link"]
    assert metadata["source"] == "news"


# --------------------------------------------------------------------------- ids


def test_chunk_id_is_stable_for_the_same_article():
    assert chunk_id(ARTICLE["link"], 0) == chunk_id(ARTICLE["link"], 0)


def test_chunk_id_separates_passages_and_articles():
    assert chunk_id(ARTICLE["link"], 0) != chunk_id(ARTICLE["link"], 1)
    assert chunk_id(ARTICLE["link"], 0) != chunk_id("https://other/article", 0)


def test_chunk_id_is_prefixed_so_news_vectors_can_be_listed():
    assert chunk_id(ARTICLE["link"], 3).startswith("news#")
    assert chunk_id(ARTICLE["link"], 3).endswith("#3")


# ------------------------------------------------------------------------ chunks


def test_chunk_documents_splits_a_long_article():
    chunks, ids = chunk_documents([_document("Le marché monte. " * 400)])

    assert len(chunks) > 1
    assert len(ids) == len(chunks)
    assert ids == [chunk_id(ARTICLE["link"], i) for i in range(len(chunks))]


def test_every_passage_carries_the_article_title():
    chunks, _ = chunk_documents([_document("Le marché monte. " * 400)])

    assert all(chunk.page_content.startswith(ARTICLE["title"]) for chunk in chunks)


def test_every_passage_carries_the_article_metadata():
    chunks, _ = chunk_documents([_document("Le marché monte. " * 400)])

    assert all(chunk.metadata["timestamp"] == 1709251200 for chunk in chunks)
    assert all(chunk.metadata["link"] == ARTICLE["link"] for chunk in chunks)


def test_a_short_article_stays_in_one_passage():
    chunks, ids = chunk_documents([_document("Le marché monte.")])

    assert len(chunks) == 1
    assert ids == [chunk_id(ARTICLE["link"], 0)]


def test_passages_stay_within_the_chunk_size():
    chunks, _ = chunk_documents([_document("Le marché monte. " * 400)])

    # The title is prepended after splitting, so allow for it on top.
    limit = CHUNK_SIZE + len(ARTICLE["title"]) + 2
    assert all(len(chunk.page_content) <= limit for chunk in chunks)


def test_an_empty_article_is_skipped():
    chunks, ids = chunk_documents([_document("   ")])

    assert chunks == []
    assert ids == []


def test_rescraping_the_same_article_reuses_its_ids():
    first, first_ids = chunk_documents([_document("Le marché monte. " * 400)])
    second, second_ids = chunk_documents([_document("Le marché monte. " * 400)])

    # Same ids means a re-run overwrites yesterday's copy instead of duplicating it.
    assert first_ids == second_ids
    assert len(first) == len(second)


# -------------------------------------------------------------------------- main


@patch("scripts.news_scraper.store_docs")
@patch("scripts.news_scraper.process_urls")
@patch("scripts.news_scraper.get_articles")
def test_main_stores_passages_with_their_ids(mock_articles, mock_process, mock_store):
    mock_articles.return_value = [ARTICLE]
    mock_process.return_value = [_document("Le marché monte. " * 400)]

    main("01/03/2024")

    docs, ids = mock_store.call_args.args
    assert len(docs) == len(ids) > 1


@patch("scripts.news_scraper.store_docs")
@patch("scripts.news_scraper.get_articles")
def test_main_stores_nothing_when_there_is_no_news(mock_articles, mock_store):
    mock_articles.return_value = []

    main("01/03/2024")

    mock_store.assert_not_called()


@patch("scripts.news_scraper.store_docs")
@patch("scripts.news_scraper.process_urls")
@patch("scripts.news_scraper.get_articles")
def test_main_stores_nothing_when_articles_are_empty(
    mock_articles, mock_process, mock_store
):
    mock_articles.return_value = [ARTICLE]
    mock_process.return_value = [_document("")]

    main("01/03/2024")

    mock_store.assert_not_called()

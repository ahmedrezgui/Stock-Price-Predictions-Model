import hashlib
import requests
from bs4 import BeautifulSoup, SoupStrainer
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import argparse

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.getcwd()) / '..'))
from src import get_pinecone_vector_store
from utils import PAGE_URL, NEWS_BASE_URL



load_dotenv()

embedding_model = os.getenv("EMBEDDING_MODEL")
index_name = os.getenv("INDEX_NAME")

DISPLAY_DATE_FORMAT = "%d/%m/%Y"
# Small enough that a passage covers about one idea, with room to keep sentences whole.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def to_timestamp(display_date: str) -> int:
    """
    Convert a dd/mm/YYYY date to epoch seconds.

    Pinecone can only compare numbers, so this is what makes a date range
    searchable; the readable date is kept alongside it.
    """
    parsed = datetime.strptime(display_date, DISPLAY_DATE_FORMAT)
    return int(parsed.replace(tzinfo=timezone.utc).timestamp())


def to_iso_date(display_date: str) -> str:
    """Convert a dd/mm/YYYY date to YYYY-MM-DD."""
    return datetime.strptime(display_date, DISPLAY_DATE_FORMAT).strftime("%Y-%m-%d")


def chunk_id(link: str, chunk_index: int) -> str:
    """
    Build the vector id for one passage of one article.

    The id is derived from the article's URL, so scraping the same day twice
    overwrites the earlier copy instead of storing a second one.
    """
    digest = hashlib.sha1(link.encode("utf-8")).hexdigest()[:16]
    return f"news#{digest}#{chunk_index}"


def build_metadata(article: dict) -> dict:
    """Assemble the metadata stored with every passage of an article."""
    return {
        "title": article["title"],
        "date": article["date"],
        "date_iso": to_iso_date(article["date"]),
        "timestamp": to_timestamp(article["date"]),
        "link": article["link"],
        "source": article["source"],
    }


def get_articles(date):
    """
    Scrape articles from the target page.

    Returns:
        articles: List of article objects with title, source, and date.

    """
    # Send request and parse
    response = requests.get(PAGE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get today's date in DD/MM/YYYY format
    if not date:
        date = datetime.now().strftime(DISPLAY_DATE_FORMAT)
        print(f"Using today's date: {date}")

    articles = []
    # Locate the specific table
    table = soup.find("table", class_="tablesorter tbl100_6 tbl3 mt37")

    if table:
        rows = table.find_all("tr")
        for row in rows:
            date_span = row.find("span", class_="sp1")
            link_tag = row.find("a", href=True)

            if date_span and link_tag:
                article_date = date_span.text.strip().split(" ")[0]
                if article_date == date:
                    article_url = link_tag["href"]
                    full_url = NEWS_BASE_URL + article_url
                    article={
                        "title": link_tag.text.strip(),
                        "link": full_url,
                        "date": article_date,
                        "source": "news"
                    }
                    articles.append(article)

    else:
        print("Articles Table not found on the page.")

    return articles


def process_urls(articles) -> list[Document]:
    """
    Load data from a list of URLs.

    Args:
        urls (Sequence): Sequence of URLs to load data from.

    Returns:
        docs: List of documents.

    """
    bs4_strainer = SoupStrainer(class_=("inarticle txtbig"))
    loader = WebBaseLoader(
        web_paths=[article["link"] for article in articles],
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.page_content = doc.page_content.replace("\n", " ").replace("\r", " ").strip().strip("Tweet")
        # Extract the title and date from the corresponding article
        doc.metadata = build_metadata(articles[i])

    return docs


def chunk_documents(docs: list[Document]) -> tuple[list[Document], list[str]]:
    """
    Split articles into passages small enough to retrieve precisely.

    Every passage keeps its article's title at the top, so one pulled out on its
    own still says which company or event it is about.

    Returns:
        The passages, and the vector id to store each one under.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks, ids = [], []
    for doc in docs:
        if not doc.page_content.strip():
            print(f"Skipping article with no readable text: {doc.metadata.get('link')}")
            continue

        title = doc.metadata.get("title")
        for index, piece in enumerate(splitter.split_text(doc.page_content)):
            chunks.append(
                Document(
                    page_content=f"{title}\n\n{piece}" if title else piece,
                    metadata=dict(doc.metadata),
                )
            )
            ids.append(chunk_id(doc.metadata["link"], index))

    return chunks, ids


def store_docs(docs, ids):
    """
    Store documents in Pinecone VectorStore.

    Args:
        docs (list[Document]): List of documents to store.
        ids (list[str]): The id to store each document under.

    """
    vector_store = get_pinecone_vector_store(index_name)

    try:
        vector_store.add_documents(docs, ids=ids)
        print(f"Stored {len(docs)} passages in Pinecone.")
    except Exception as e:
        print(f"Error storing documents in Pinecone: {e}")



def main(date):
    articles = get_articles(date)

    if len(articles) == 0:
        print("No articles found for today.")
        return

    docs = process_urls(articles)
    chunks, ids = chunk_documents(docs)

    if not chunks:
        print("None of the articles had readable text.")
        return

    store_docs(chunks, ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and Storing automation for news articles.")
    parser.add_argument("--date", required=False, help="Articles date in dd/mm/YYYY format")
    args = parser.parse_args()
    main(args.date)

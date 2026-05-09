"""
scrapers/base.py — Base scraper interface.
All scrapers inherit from this. Keeps ingest pipeline consistent.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup


RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Restrict HTTP fetches to known government / official domains.
# Prevents SSRF if a scraped page issues an open redirect.
ALLOWED_DOMAINS = {
    "cbic.gov.in",
    "cbic-gst.gov.in",
    "gstcouncil.gov.in",
    "indiabudget.gov.in",
    "icai.org",
    "ficci.in",
    "loksabha.nic.in",
    "rajyasabha.nic.in",
    "eci.gov.in",
}

# Cap response body to 10 MB — guards against runaway/misconfigured servers.
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class Document:
    """A scraped document, normalised before processing."""

    def __init__(
        self,
        source_id: str,
        doc_id: str,
        title: str,
        url: str,
        date: Optional[datetime],
        content: str,
        metadata: dict = None,
    ):
        self.source_id = source_id
        self.doc_id = doc_id
        self.title = title
        self.url = url
        self.date = date
        self.content = content
        self.metadata = metadata or {}
        self.scraped_at = datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "url": self.url,
            "date": self.date.isoformat() if self.date else None,
            "content": self.content,
            "metadata": self.metadata,
            "scraped_at": self.scraped_at,
        }

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class BaseScraper(ABC):
    """
    All scrapers follow this interface.
    Implement `scrape()` — return list of Document objects.
    The ingest pipeline handles dedup, storage, and triggering processors.
    """

    source_id: str  # must match key in config/sources.yaml

    def __init__(self):
        self.output_dir = RAW_DATA_DIR / self.source_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            timeout=30,
            headers={"User-Agent": "gst-foresight/1.0 (research; contact via github)"},
            follow_redirects=True,
        )

    @abstractmethod
    def scrape(self) -> list[Document]:
        """Fetch and return documents. Don't filter for duplicates here."""
        pass

    def save(self, docs: list[Document]) -> int:
        """Save documents, skip already-seen content. Returns count of new docs."""
        new_count = 0
        for doc in docs:
            path = self.output_dir / f"{doc.doc_id}.json"
            if not path.exists():
                path.write_text(json.dumps(doc.to_dict(), indent=2, default=str))
                new_count += 1
        return new_count

    def _validate_url(self, url: str) -> None:
        """Raise ValueError if the URL's domain isn't in the allowlist."""
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        # Check exact match or subdomain match (e.g. sub.cbic.gov.in)
        if not any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS):
            raise ValueError(f"fetch blocked — domain not in allowlist: {host!r}")

    def fetch_html(self, url: str) -> BeautifulSoup:
        self._validate_url(url)
        r = self.client.get(url)
        r.raise_for_status()
        # Guard against oversized responses before parsing
        if len(r.content) > MAX_RESPONSE_BYTES:
            raise ValueError(
                f"response too large ({len(r.content):,} bytes) for {url!r}"
            )
        return BeautifulSoup(r.text, "html.parser")

    def __del__(self):
        self.client.close()

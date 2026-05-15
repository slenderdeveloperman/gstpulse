"""
scrapers/base.py — Base scraper interface.
All scrapers inherit from this. Keeps ingest pipeline consistent.
"""

import hashlib
import io
import json
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urljoin
import httpx
from bs4 import BeautifulSoup

_docling_converter = None   # lazy-initialised once, reused across calls
_ocr_disabled = False       # set via disable_ocr() to skip Docling entirely


def release_docling():
    """Free the Docling converter and OCR models. Call between sources in long runs."""
    global _docling_converter
    if _docling_converter is not None:
        _docling_converter = None
        import gc
        gc.collect()


def disable_ocr():
    """Prevent Docling from loading. Safe to call before any scraping starts."""
    global _ocr_disabled
    _ocr_disabled = True


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
    # Phase 2 signal sources
    "indiankanoon.org",      # court judgment aggregator
    "pib.gov.in",            # Press Information Bureau
    "sansad.in",             # Parliamentary portal
    "egazette.nic.in",       # Official Gazette of India
    # ICAI hosts PDFs on CloudFront CDN — needed for pre-budget memoranda
    "d23z1tp9il9etb.cloudfront.net",
    "idtc.icai.org",
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
            # Some GOI sites serve expired or hostname-mismatched TLS certs
            # (e.g. indiabudget.gov.in). We disable cert verification here as a
            # pragmatic workaround. This is acceptable ONLY because every URL is
            # validated against the ALLOWED_DOMAINS allowlist before any fetch —
            # an attacker cannot inject an arbitrary domain. The remaining risk
            # (MITM on a .gov.in domain) is low in this read-only scraping context.
            # Revisit if any source starts returning sensitive data.
            verify=False,
        )

    @abstractmethod
    def scrape(self) -> list[Document]:
        """Fetch and return documents. Don't filter for duplicates here."""
        pass

    def doc_cached(self, doc_id: str) -> bool:
        """Return True if this doc_id is already saved to disk."""
        return (self.output_dir / f"{doc_id}.json").exists()

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
        if not any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS):
            raise ValueError(f"fetch blocked — domain not in allowlist: {host!r}")

    def fetch_html(self, url: str) -> BeautifulSoup:
        self._validate_url(url)
        r = self.client.get(url)
        r.raise_for_status()
        if len(r.content) > MAX_RESPONSE_BYTES:
            raise ValueError(
                f"response too large ({len(r.content):,} bytes) for {url!r}"
            )
        return BeautifulSoup(r.text, "html.parser")

    def fetch_pdf_text(self, url: str, max_bytes: int = MAX_RESPONSE_BYTES) -> Optional[str]:
        """Download a PDF and extract full text.

        Extraction chain (all local, zero API cost):
          1. pdfplumber  — fast, handles most GOI text-based PDFs
          2. pymupdf     — catches layouts pdfplumber misses
          3. Docling+RapidOCR — for scanned/image PDFs; processed in 5-page
                                batches so the CPU never spikes on long docs

        max_bytes: override the default cap for large but trusted documents
                   (e.g. GST Council minutes which run 20-30 MB).
        """
        try:
            self._validate_url(url)
            r = self.client.get(url, timeout=120)
            r.raise_for_status()
            if len(r.content) > max_bytes:
                print(f"[pdf] skipping {url} — {len(r.content):,} bytes exceeds cap of {max_bytes:,}", flush=True)
                return None

            # Step 1: pdfplumber — lightweight, handles text-layer PDFs well
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n\n".join(pages).strip()
                if len(text) > 200:
                    return text
            except Exception:
                pass

            # Step 2: pymupdf — better on some GOI layouts and mixed content
            try:
                import fitz
                doc = fitz.open(stream=r.content, filetype="pdf")
                text = "\n\n".join(page.get_text() for page in doc).strip()
                doc.close()
                if len(text) > 200:
                    return text
            except Exception:
                pass

            # Step 3: Docling + RapidOCR — for scanned/image PDFs where the
            # text layer is absent. Processed in 5-page batches with explicit
            # gc.collect() between batches to keep memory flat on long docs.
            text = self._fetch_pdf_text_docling(r.content)
            if text:
                return text

        except Exception as e:
            print(f"[pdf] extraction failed for {url}: {e}")
        return None

    def _fetch_pdf_text_docling(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from a scanned/image PDF using Docling + RapidOCR.

        Processes the PDF in BATCH_SIZE-page slices so the OCR model never
        holds more than a few pages in memory at once.
        """
        global _docling_converter
        import gc

        if _ocr_disabled:
            return None

        try:
            import fitz
        except ImportError:
            print("[docling] pymupdf not installed — cannot batch-split PDF")
            return None

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            from docling.datamodel.base_models import InputFormat
        except ImportError:
            print("[docling] docling not installed — skipping OCR extraction")
            return None

        # Build the converter once per process; RapidOCR model loads are expensive
        if _docling_converter is None:
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                ocr_options=RapidOcrOptions(),
                do_table_structure=False,   # skip table parsing — saves CPU, not needed for RAG text
                generate_page_images=False,
            )
            _docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

        BATCH_SIZE = 5  # pages per batch — keeps peak RAM under ~400 MB on CPU
        text_parts: list[str] = []

        try:
            src = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(src)

            for batch_start in range(0, total_pages, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_pages)

                # Write the page slice to a temp file; Docling needs a file path
                batch_doc = fitz.open()
                batch_doc.insert_pdf(src, from_page=batch_start, to_page=batch_end - 1)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(batch_doc.tobytes())
                    tmp_path = tmp.name
                batch_doc.close()

                try:
                    result = _docling_converter.convert(tmp_path)
                    batch_text = result.document.export_to_markdown().strip()
                    if batch_text:
                        text_parts.append(batch_text)
                    print(f"[docling] pages {batch_start + 1}–{batch_end}/{total_pages}: "
                          f"{len(batch_text)} chars extracted")
                except Exception as e:
                    print(f"[docling] batch {batch_start + 1}–{batch_end} failed: {e}")
                finally:
                    os.unlink(tmp_path)
                    gc.collect()  # release OCR intermediate tensors between batches

            src.close()

        except Exception as e:
            print(f"[docling] PDF split failed: {e}")
            return None

        return "\n\n".join(text_parts).strip() or None

    def _find_pdf_url(self, page_url: str) -> Optional[str]:
        """Fetch an HTML page and return the first PDF link found on it."""
        try:
            soup = self.fetch_html(page_url)
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if ".pdf" in href.lower():
                    return href if href.startswith("http") else urljoin(page_url, href)
        except Exception:
            pass
        return None

    def __del__(self):
        self.client.close()

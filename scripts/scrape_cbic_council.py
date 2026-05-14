"""
scripts/scrape_cbic_council.py — Scrapes CBIC circulars + GST Council minutes,
then tags, chunks, and embeds the new documents into Supabase.

Memory guardrails:
  - OCR (Docling + RapidOCR) is disabled — CBIC and GST Council PDFs are
    text-based; pdfplumber handles them without loading any ML models.
  - Memory is checked before each document fetch and before each embed batch.
  - Hard limits: warn at <1.5 GB free, skip document at <800 MB free.

Sarvam note:
  - This script does NOT call Sarvam at any point.
  - The only external API call is to the Supabase gte-small edge function
    for embeddings. If you see repeated embed failures, check the Supabase
    dashboard — not your Sarvam account.
"""

import gc
import json
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.base import disable_ocr, release_docling
from scrapers.sources import CBICCircularScraper, GSTCouncilScraper
from processors.tagger import TopicTagger
from processors.chunker import Chunker
from processors.embedder import Embedder

# ── Memory thresholds ─────────────────────────────────────────────────────────
MEM_WARN_MB   = 1_500   # log a warning below this much free RAM
MEM_SKIP_MB   =   800   # skip the current document below this


def free_mb() -> float:
    return psutil.virtual_memory().available / 1024 / 1024


def mem_log(label: str) -> None:
    vm = psutil.virtual_memory()
    free = vm.available / 1024 / 1024
    used = vm.used / 1024 / 1024
    flag = " ⚠ LOW" if free < MEM_WARN_MB else ""
    print(f"[mem] {label}: {free:.0f} MB free / {used:.0f} MB used{flag}", flush=True)


def check_mem_or_skip(doc_id: str) -> bool:
    """Return False and log a warning if memory is too low to safely process this doc."""
    free = free_mb()
    if free < MEM_SKIP_MB:
        print(
            f"[mem] SKIP {doc_id} — only {free:.0f} MB free (<{MEM_SKIP_MB} MB threshold). "
            f"Run again after other processes free memory.",
            flush=True,
        )
        return False
    if free < MEM_WARN_MB:
        print(f"[mem] WARNING — {free:.0f} MB free before processing {doc_id}", flush=True)
    return True


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def tag_new_docs(tagger: TopicTagger, source_id: str) -> int:
    raw_dir = Path("data/raw") / source_id
    tagged = 0
    for raw_path in sorted(raw_dir.glob("*.json")):
        processed_path = Path("data/processed") / raw_path.name
        if processed_path.exists():
            continue
        if not check_mem_or_skip(raw_path.name):
            continue
        result = tagger.tag_and_save(raw_path)
        if result:
            tagged += 1
            print(f"[tagger] tagged {raw_path.name} → topics: {result.get('topic_tags', [])}", flush=True)
    return tagged


def chunk_new_docs(chunker: Chunker, source_id: str) -> list[Path]:
    """Chunk any unprocessed docs for source_id. Returns paths of new chunk files."""
    new_chunk_paths = []
    for processed_path in sorted(Path("data/processed").glob("*.json")):
        if chunker.chunks_exist(processed_path):
            continue
        if not check_mem_or_skip(processed_path.name):
            continue
        chunks = chunker.chunk_and_save(processed_path)
        if chunks:
            new_chunk_paths.append(Path("data/chunks") / processed_path.name)
            print(f"[chunker] {processed_path.name} → {len(chunks)} chunks", flush=True)
    return new_chunk_paths


def embed_new_chunks(embedder: Embedder, chunk_paths: list[Path]) -> int:
    """Embed and upsert chunks for the given chunk file paths."""
    total = 0
    for chunk_path in chunk_paths:
        if not chunk_path.exists():
            continue
        mem_log(f"before embedding {chunk_path.name}")
        if not check_mem_or_skip(chunk_path.name):
            continue
        chunks = json.loads(chunk_path.read_text())
        if not chunks:
            continue
        try:
            indexed = embedder.embed_chunks(chunks)
            total += indexed
            print(f"[embed] {chunk_path.name} → {indexed} vectors upserted", flush=True)
        except Exception as e:
            print(f"[embed] FAILED {chunk_path.name}: {e} — skipping, will retry next run", flush=True)
        gc.collect()
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def run_source(source_id: str, scraper_cls, tagger: TopicTagger, chunker: Chunker):
    print(f"\n{'='*60}", flush=True)
    print(f"SOURCE: {source_id}", flush=True)
    print(f"{'='*60}", flush=True)
    mem_log("start")

    # 1. Scrape
    print(f"\n[scrape] starting {source_id}...", flush=True)
    scraper = scraper_cls()
    docs = scraper.scrape()
    new_count = scraper.save(docs)
    print(f"[scrape] {source_id}: {len(docs)} found, {new_count} new", flush=True)
    del docs, scraper
    release_docling()
    gc.collect()
    mem_log("after scrape")

    # 2. Tag
    print(f"\n[tag] tagging new docs for {source_id}...", flush=True)
    tagged = tag_new_docs(tagger, source_id)
    print(f"[tag] {source_id}: {tagged} new docs tagged", flush=True)
    gc.collect()
    mem_log("after tagging")

    # 3. Chunk
    print(f"\n[chunk] chunking new docs for {source_id}...", flush=True)
    new_chunk_paths = chunk_new_docs(chunker, source_id)
    print(f"[chunk] {source_id}: {len(new_chunk_paths)} new chunk files", flush=True)
    gc.collect()
    mem_log("after chunking")

    return new_chunk_paths


def main():
    print("[scrape_cbic_council] starting — OCR disabled, memory guardrails active", flush=True)
    print("[scrape_cbic_council] NOTE: NO Sarvam API calls in this script. "
          "Embeddings use Supabase gte-small only.", flush=True)
    mem_log("initial")

    # Disable Docling/RapidOCR — CBIC and GST Council PDFs are text-based.
    # pdfplumber handles them; loading OCR models would waste ~1.5 GB RAM.
    disable_ocr()

    tagger  = TopicTagger()
    chunker = Chunker()

    sources = [
        ("cbic_circulars",     CBICCircularScraper),
        ("gst_council_minutes", GSTCouncilScraper),
    ]

    all_new_chunk_paths = []
    for source_id, scraper_cls in sources:
        chunk_paths = run_source(source_id, scraper_cls, tagger, chunker)
        all_new_chunk_paths.extend(chunk_paths)
        gc.collect()

    # 4. Embed all new chunks in one pass
    if not all_new_chunk_paths:
        print("\n[embed] no new chunks to embed — all docs already indexed", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"EMBED: {len(all_new_chunk_paths)} new chunk files", flush=True)
    print(f"{'='*60}", flush=True)
    mem_log("before embed pass")

    embedder = Embedder()
    total_indexed = embed_new_chunks(embedder, all_new_chunk_paths)
    del embedder
    gc.collect()

    mem_log("after embed pass")
    print(f"\n[scrape_cbic_council] done — {total_indexed} new vectors indexed to Supabase", flush=True)
    print("[scrape_cbic_council] run: python -m gst_foresight predict", flush=True)


if __name__ == "__main__":
    main()

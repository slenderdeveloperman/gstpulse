"""
scripts/scrape_pib.py — Scrapes PIB Finance Ministry press releases,
then tags and chunks new documents. Pass --skip-embed to skip Supabase
upsert (useful for dry runs or when the embed quota is exhausted).

OCR is disabled — PIB press releases are text HTML pages, not scanned PDFs.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.base import disable_ocr, release_docling
from scrapers.sources import PIBFinanceScraper
from processors.tagger import TopicTagger
from processors.chunker import Chunker

MEM_WARN_MB = 1_500
MEM_SKIP_MB =   800


def free_mb() -> float:
    return psutil.virtual_memory().available / 1024 / 1024


def mem_log(label: str) -> None:
    vm = psutil.virtual_memory()
    free = vm.available / 1024 / 1024
    used = vm.used / 1024 / 1024
    flag = " ⚠ LOW" if free < MEM_WARN_MB else ""
    print(f"[mem] {label}: {free:.0f} MB free / {used:.0f} MB used{flag}", flush=True)


def check_mem_or_skip(doc_id: str) -> bool:
    free = free_mb()
    if free < MEM_SKIP_MB:
        print(
            f"[mem] SKIP {doc_id} — only {free:.0f} MB free (<{MEM_SKIP_MB} MB threshold)",
            flush=True,
        )
        return False
    if free < MEM_WARN_MB:
        print(f"[mem] WARNING — {free:.0f} MB free before processing {doc_id}", flush=True)
    return True


def tag_new_docs(tagger: TopicTagger, source_id: str) -> int:
    raw_dir = Path("data/raw") / source_id
    tagged = 0
    files = sorted(raw_dir.glob("*.json"))
    print(f"[tagger] scanning {len(files)} raw files for {source_id}...", flush=True)
    for raw_path in files:
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


def chunk_new_docs(chunker: Chunker) -> list[Path]:
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


def embed_new_chunks(chunk_paths: list[Path]) -> int:
    from processors.embedder import Embedder
    embedder = Embedder()
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


def main():
    parser = argparse.ArgumentParser(description="Scrape PIB Finance press releases")
    parser.add_argument("--skip-embed", action="store_true", help="skip Supabase embedding step")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("PIB Finance scraper", flush=True)
    print("Strategy: RSS anchor → PRID enumeration (step 25, last 90 days)", flush=True)
    print(f"Embed: {'SKIPPED (--skip-embed)' if args.skip_embed else 'ENABLED'}", flush=True)
    print("=" * 60, flush=True)
    mem_log("initial")

    disable_ocr()

    # ── 1. Scrape ────────────────────────────────────────────────────────────
    print("\n[scrape] starting pib_finance...", flush=True)
    t0 = time.time()
    scraper = PIBFinanceScraper()
    docs = scraper.scrape()
    new_count = scraper.save(docs)
    elapsed = time.time() - t0
    print(f"[scrape] done in {elapsed:.1f}s — {len(docs)} found, {new_count} new saved to disk", flush=True)
    del docs, scraper
    release_docling()
    gc.collect()
    mem_log("after scrape")

    if new_count == 0:
        print("\n[scrape] no new documents — nothing to tag/chunk/embed. Done.", flush=True)
        return

    # ── 2. Tag ───────────────────────────────────────────────────────────────
    print("\n[tag] tagging new pib_finance docs...", flush=True)
    tagger = TopicTagger()
    tagged = tag_new_docs(tagger, "pib_finance")
    print(f"[tag] {tagged} new docs tagged", flush=True)
    del tagger
    gc.collect()
    mem_log("after tagging")

    # ── 3. Chunk ─────────────────────────────────────────────────────────────
    print("\n[chunk] chunking new docs...", flush=True)
    chunker = Chunker()
    new_chunk_paths = chunk_new_docs(chunker)
    print(f"[chunk] {len(new_chunk_paths)} new chunk files created", flush=True)
    del chunker
    gc.collect()
    mem_log("after chunking")

    # ── 4. Embed ─────────────────────────────────────────────────────────────
    if args.skip_embed:
        print("\n[embed] skipped (--skip-embed flag)", flush=True)
    elif new_chunk_paths:
        print(f"\n[embed] embedding {len(new_chunk_paths)} chunk files into Supabase...", flush=True)
        total_vectors = embed_new_chunks(new_chunk_paths)
        print(f"[embed] done — {total_vectors} vectors upserted total", flush=True)
        mem_log("after embedding")
    else:
        print("\n[embed] no new chunk files to embed", flush=True)

    print("\n[done] PIB Finance ingest complete.", flush=True)


if __name__ == "__main__":
    main()

"""
scripts/embed_council_chunks.py — Embed all GST Council meeting chunk files into Supabase.

Designed to be re-runnable: Supabase upsert is idempotent, so already-indexed
chunks are safely overwritten with identical data. Run this after a rate-limit
crash to pick up where the previous run failed.

Rate-limit guardrail: embedder.py sleeps 150ms between calls (~6 calls/sec).
Error handling: per-file failures are logged and skipped; the run continues.
"""

import gc
import json
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.embedder import Embedder

CHUNKS_DIR = Path("data/chunks")
MEM_WARN_MB = 1_500
MEM_SKIP_MB = 800


def free_mb() -> float:
    return psutil.virtual_memory().available / 1024 / 1024


def mem_log(label: str) -> None:
    vm = psutil.virtual_memory()
    free = vm.available / 1024 / 1024
    used = vm.used / 1024 / 1024
    flag = " ⚠ LOW" if free < MEM_WARN_MB else ""
    print(f"[mem] {label}: {free:.0f} MB free / {used:.0f} MB used{flag}", flush=True)


def main():
    chunk_files = sorted(CHUNKS_DIR.glob("gst_council_*.json"))
    print(f"[embed_council] {len(chunk_files)} chunk files to embed", flush=True)
    print("[embed_council] NOTE: embedder uses Supabase gte-small. Zero Sarvam calls.", flush=True)
    mem_log("start")

    embedder = Embedder()
    total = 0
    failed = []

    for i, chunk_path in enumerate(chunk_files, 1):
        free = free_mb()
        if free < MEM_SKIP_MB:
            print(f"[mem] SKIP {chunk_path.name} — only {free:.0f} MB free", flush=True)
            continue
        if free < MEM_WARN_MB:
            print(f"[mem] WARNING — {free:.0f} MB free before {chunk_path.name}", flush=True)

        chunks = json.loads(chunk_path.read_text())
        if not chunks:
            print(f"[{i}/{len(chunk_files)}] {chunk_path.name} — empty, skipping", flush=True)
            continue

        print(f"[{i}/{len(chunk_files)}] {chunk_path.name} — {len(chunks)} chunks", flush=True)
        try:
            indexed = embedder.embed_chunks(chunks)
            total += indexed
            print(f"  → {indexed} vectors upserted (running total: {total})", flush=True)
        except Exception as e:
            print(f"  → FAILED: {e} — logged, continuing", flush=True)
            failed.append(chunk_path.name)

        gc.collect()

    mem_log("done")
    print(f"\n[embed_council] finished — {total} vectors upserted total", flush=True)
    if failed:
        print(f"[embed_council] {len(failed)} files failed: {failed}", flush=True)
        print("[embed_council] re-run this script to retry failed files", flush=True)
    else:
        print("[embed_council] all files embedded successfully", flush=True)

    print("[embed_council] run: python -m gst_foresight predict", flush=True)


if __name__ == "__main__":
    main()

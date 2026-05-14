"""Minimal standalone chunking script — no scrapers, no chromadb, no heavy deps."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.chunker import Chunker

processed_dir = Path("data/processed")
chunks_dir = Path("data/chunks")

chunker = Chunker()
paths = [p for p in sorted(processed_dir.glob("*.json")) if not chunker.chunks_exist(p)]

print(f"[chunk_all] {len(paths)} docs to chunk", flush=True)

done = skipped = failed = 0
total_chunks = 0

for i, path in enumerate(paths, 1):
    print(f"\n[chunk_all] [{i}/{len(paths)}] {path.name}", flush=True)
    t0 = time.monotonic()
    chunks = chunker.chunk_and_save(path)
    elapsed = time.monotonic() - t0

    if chunks:
        done += 1
        total_chunks += len(chunks)
        print(f"[chunk_all] ✓ {len(chunks)} chunks in {elapsed:.2f}s", flush=True)
    else:
        skipped += 1
        print(f"[chunk_all] - skipped (no content) in {elapsed:.2f}s", flush=True)

print(f"\n[chunk_all] finished — {done} chunked ({total_chunks} total chunks), {skipped} skipped, {failed} failed")
print(f"[chunk_all] next: python -m gst_foresight embed")

"""
scripts/test_chunk_embed.py — Chunk and embed 2 CBIC circulars via Supabase gte-small.

Run from the project root:
    python scripts/test_chunk_embed.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.chunker import Chunker
from processors.embedder import Embedder

PROCESSED_DIR = Path("data/processed")

TEST_DOCS = [
    "cbic_circ_pdf_circular_no_249_2025_pdf.json",
    "cbic_circ_pdf_circular_no_250_2025_pdf.json",
]


def main():
    chunker = Chunker()
    embedder = Embedder()

    for filename in TEST_DOCS:
        path = PROCESSED_DIR / filename
        if not path.exists():
            print(f"[skip] {filename} — not found in data/processed/")
            continue

        doc = json.loads(path.read_text())
        content_len = len(doc.get("content") or "")
        print(f"\n── {filename}")
        print(f"   title  : {doc.get('title', '(no title)')[:80]}")
        print(f"   content: {content_len:,} chars")

        chunks = chunker.chunk(doc)
        print(f"   chunks : {len(chunks)}")

        if not chunks:
            print("   [skip] no content to embed")
            continue

        # Save chunk file so status command picks it up
        chunk_path = Path("data/chunks") / filename
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_text(json.dumps(chunks, indent=2, default=str))
        print(f"   saved  : data/chunks/{filename}")

        new_vecs = embedder.embed_chunks(chunks)
        print(f"   indexed: {new_vecs} new vectors")

    stats = embedder.stats()
    print(f"\nTotal in Supabase: {stats['total_chunks']} chunks ({stats['model']})")


if __name__ == "__main__":
    main()

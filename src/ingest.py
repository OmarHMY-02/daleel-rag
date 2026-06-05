"""Ingestion: docs -> markdown-aware chunks -> E5 embeddings -> Chroma.

Run directly to (re)build a vector store for a given chunking config:

    python -m src.ingest --chunk-size 500 --overlap 75
    python -m src.ingest --chunk-size 300 --overlap 60

Each config is stored in its own Chroma collection (see ``collection_name``)
so the evaluation can compare configs against a matching index. The persisted
store under ``chroma/`` is committed, so the deployed app needs no build step.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, List

from .config import CONFIG, DOCS_DIR, collection_name
from .providers import Embedder

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


# --------------------------------------------------------------------------- #
# Markdown-aware chunking
# --------------------------------------------------------------------------- #
def _split_sections(text: str) -> List[Dict[str, str]]:
    """Split a markdown doc into (title, heading, body) sections.

    ``title`` is the document's top-level ``#`` heading (kept on every chunk so
    a citation reads e.g. "Leave & Attendance Policy — Annual Leave").
    ``heading`` is the most recent section heading. Blockquote lines (``>``) are
    dropped — in this corpus they are disclaimers, not policy facts.
    """
    title = ""
    heading = "Overview"
    buf: List[str] = []
    sections: List[Dict[str, str]] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append({"title": title, "heading": heading, "body": body})
        buf.clear()

    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            text_h = m.group(2).strip()
            if level == 1 and not title:
                title = text_h
                # strip a trailing "— Something (Year)" tail into a clean title
                continue
            flush()
            heading = text_h
            continue
        if line.lstrip().startswith(">"):
            continue
        buf.append(line)
    flush()
    return sections


def _paragraph_units(body: str) -> List[str]:
    """Body -> list of packing units: paragraphs, with over-long paragraphs
    further split into sentences so no single unit blows the token budget."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    units: List[str] = []
    for p in paras:
        # A bullet list collapses to one line per item; keep items whole.
        if len(p) > 600:  # likely needs sentence-level splitting
            for sent in re.split(r"(?<=[.!?؟])\s+", p):
                sent = sent.strip()
                if sent:
                    units.append(sent)
        else:
            units.append(p)
    return units


def _pack(units: List[str], chunk_size: int, overlap: int,
          token_len: Callable[[str], int]) -> List[str]:
    """Greedily pack units into <= chunk_size token chunks with a token-measured
    overlap tail carried into the next chunk (preserves cross-boundary context)."""
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0

    for unit in units:
        ut = token_len(unit)
        if cur and cur_tok + ut > chunk_size:
            chunks.append("\n".join(cur))
            # seed next chunk with the trailing units that fit in `overlap`
            seed: List[str] = []
            seed_tok = 0
            for u in reversed(cur):
                tu = token_len(u)
                if seed_tok + tu > overlap:
                    break
                seed.insert(0, u)
                seed_tok += tu
            cur, cur_tok = list(seed), seed_tok
        cur.append(unit)
        cur_tok += ut

    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_markdown(text: str, source: str, chunk_size: int, overlap: int,
                   token_len: Callable[[str], int]) -> List[Dict]:
    """Return chunk dicts with citation metadata (source file + section heading)."""
    out: List[Dict] = []
    for sec in _split_sections(text):
        units = _paragraph_units(sec["body"])
        for piece in _pack(units, chunk_size, overlap, token_len):
            out.append(
                {
                    "text": piece,
                    "source": source,
                    "title": sec["title"],
                    "heading": sec["heading"],
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def get_client():
    import chromadb

    return chromadb.PersistentClient(path=CONFIG.chroma_dir)


def get_collection(chunk_size: int, overlap: int):
    """Open an existing collection for a chunking config (read path for the app)."""
    client = get_client()
    return client.get_collection(collection_name(chunk_size, overlap))


def build_index(chunk_size: int, overlap: int, embedder: Embedder | None = None,
                verbose: bool = True) -> int:
    """(Re)build the Chroma collection for one chunking config. Returns #chunks."""
    embedder = embedder or Embedder()
    client = get_client()
    name = collection_name(chunk_size, overlap)

    # Fresh build every time so re-running is deterministic.
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []
    embed_inputs: List[str] = []

    for path in sorted(Path(DOCS_DIR).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(raw, path.name, chunk_size, overlap, embedder.token_len)
        for i, ch in enumerate(chunks):
            ids.append(f"{path.name}::{i}")
            documents.append(ch["text"])
            metadatas.append(
                {"source": ch["source"], "title": ch["title"], "heading": ch["heading"]}
            )
            # Embed heading + text so section context is captured in the vector.
            embed_inputs.append(f"{ch['title']} — {ch['heading']}\n{ch['text']}")
        if verbose:
            print(f"  {path.name}: {len(chunks)} chunks")

    embeddings = embedder.embed_documents(embed_inputs)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    if verbose:
        print(f"Built '{name}' with {len(ids)} chunks at {CONFIG.chroma_dir}")
    return len(ids)


def main():
    ap = argparse.ArgumentParser(description="Build the Daleel vector store.")
    ap.add_argument("--chunk-size", type=int, default=CONFIG.chunk_size)
    ap.add_argument("--overlap", type=int, default=CONFIG.chunk_overlap)
    args = ap.parse_args()
    build_index(args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()

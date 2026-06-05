"""Retrieval-augmented generation: retrieve top-k chunks, then answer with
citations — grounded strictly in the retrieved context.

The prompt is the heart of the trustworthiness story: the model is told to use
only the context, to refuse with an exact phrase when the context is
insufficient (this is what makes the unanswerable gold question work), and to
cite the source filenames it used. ``answer()`` returns both the text and the
retrieved chunks so the UI can show citations and the eval can score grounding.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import CONFIG
from .ingest import get_collection
from .providers import Embedder, generate

# Exact phrase the model must use when it cannot answer. The eval detects this
# string to score refusal behaviour, so it must stay in sync with metrics.py.
REFUSAL_TEXT = "I cannot answer this from the available documents."

SYSTEM_PROMPT = (
    "You are Daleel, a careful assistant for Najmah Retail Group. "
    "Answer the user's question using ONLY the information in the provided "
    "context passages. Rules:\n"
    "1. Base every statement only on the context. Never use outside knowledge "
    "or guess.\n"
    f'2. If the context does not contain enough information, reply with exactly: '
    f'"{REFUSAL_TEXT}" and nothing else.\n'
    "3. When you answer, be concise and include the exact figures from the "
    "context.\n"
    "4. End with a line starting 'Sources:' listing the filenames of the "
    "passages you actually used."
)


def _embedder() -> Embedder:
    # Cached inside providers via lru_cache; cheap to re-instantiate.
    return Embedder()


def retrieve(question: str, k: Optional[int] = None,
             chunk_size: Optional[int] = None,
             overlap: Optional[int] = None) -> List[Dict]:
    """Top-k cosine retrieval over the matching Chroma collection."""
    k = k or CONFIG.top_k
    chunk_size = chunk_size or CONFIG.chunk_size
    overlap = overlap if overlap is not None else CONFIG.chunk_overlap

    collection = get_collection(chunk_size, overlap)
    q_emb = _embedder().embed_query(question)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    out: List[Dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": doc,
                "source": meta.get("source", "?"),
                "heading": meta.get("heading", ""),
                "title": meta.get("title", ""),
                # cosine distance -> similarity for a friendly 0..1 score
                "score": round(1.0 - float(dist), 4),
            }
        )
    return out


def build_prompt(question: str, contexts: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(
            f"[{i}] (source: {c['source']} — {c['heading']})\n{c['text']}"
        )
    context_str = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return (
        f"Context passages:\n\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def answer(question: str, k: Optional[int] = None,
           chunk_size: Optional[int] = None, overlap: Optional[int] = None,
           backend: Optional[str] = None, api_key: Optional[str] = None) -> Dict:
    """Retrieve + generate. Returns ``{answer, contexts}`` where ``contexts`` is
    the list of retrieved chunks (for citations and grounding eval)."""
    contexts = retrieve(question, k=k, chunk_size=chunk_size, overlap=overlap)
    prompt = build_prompt(question, contexts)
    text = generate(
        prompt,
        system=SYSTEM_PROMPT,
        temperature=0.0,
        backend=backend,
        api_key=api_key,
    )
    return {"answer": (text or "").strip(), "contexts": contexts}

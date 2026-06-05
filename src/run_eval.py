"""Run the RAG pipeline over the gold question set and write a scored eval run.

    python -m src.run_eval --chunk-size 500 --overlap 75 --k 4
    python -m src.run_eval --chunk-size 300 --overlap 60 --k 4

Writes ``eval/eval_runs/<timestamp>_<config>.json`` with per-question scores,
the retrieved chunks, the generated answer, and aggregate metrics. Commit two
runs with different configs so the dashboard's comparison view has real data.
The judge uses the same free backend, so a full run costs $0.
"""

from __future__ import annotations

import argparse
import functools
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import metrics
from .config import CONFIG, EVAL_RUNS_DIR, GOLD_QA_PATH, collection_name
from .ingest import build_index, get_client
from .providers import Embedder, backend_ready, generate
from .rag import answer as rag_answer


def _ensure_index(chunk_size: int, overlap: int, embedder: Embedder) -> None:
    """Build the collection for this config if it isn't already present."""
    name = collection_name(chunk_size, overlap)
    # Chroma >= 0.6 returns collection *names* (strings) from list_collections().
    existing = set(get_client().list_collections())
    if name not in existing:
        print(f"[index] building missing collection {name} ...")
        build_index(chunk_size, overlap, embedder=embedder, verbose=True)


def run(chunk_size: int, overlap: int, k: int,
        backend: Optional[str] = None) -> Dict:
    backend = (backend or CONFIG.llm_backend).lower()
    ok, status = backend_ready(backend)
    if not ok:
        raise SystemExit(f"Generation backend not ready: {status}")

    model_name = CONFIG.groq_model if backend == "groq" else CONFIG.ollama_model
    print(f"[eval] backend={backend} model={model_name} "
          f"chunk={chunk_size} overlap={overlap} k={k}")

    embedder = Embedder()
    _ensure_index(chunk_size, overlap, embedder)
    judge = functools.partial(generate, backend=backend)

    gold = json.loads(GOLD_QA_PATH.read_text(encoding="utf-8"))
    results: List[Dict] = []

    for item in gold:
        out = rag_answer(item["question"], k=k, chunk_size=chunk_size, overlap=overlap,
                         backend=backend)
        ans, contexts = out["answer"], out["contexts"]

        expect_refusal = (item["type"] == "unanswerable") or not item["source_docs"]
        refused = metrics.is_refusal(ans)

        faith = None if refused else metrics.faithfulness(ans, contexts, judge)
        rel = None if refused else metrics.answer_relevance(
            item["question"], metrics.strip_sources_line(ans), embedder)
        cprec = metrics.context_precision(contexts, item["source_docs"])
        rcorrect = metrics.refusal_correct(ans, expect_refusal)

        print(f"  {item['id']:<4} {item['type']:<16} "
              f"faith={faith} rel={rel} ctx_prec={cprec} refusal_ok={rcorrect}")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "type": item["type"],
            "ground_truth": item["ground_truth"],
            "gold_source_docs": item["source_docs"],
            "answer": ans,
            "refused": refused,
            "expected_refusal": expect_refusal,
            "retrieved": [
                {"source": c["source"], "heading": c["heading"],
                 "score": c["score"], "text": c["text"]}
                for c in contexts
            ],
            "metrics": {
                "faithfulness": faith,
                "answer_relevance": rel,
                "context_precision": cprec,
                "refusal_correct": rcorrect,
            },
        })

    aggregate = _aggregate(results)
    run_obj = {
        "config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "k": k,
            "backend": backend,
            "model": model_name,
            "embed_model": CONFIG.embed_model,
            "label": f"chunk{chunk_size}_k{k}",
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aggregate": aggregate,
        "results": results,
    }

    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = EVAL_RUNS_DIR / f"{ts}_chunk{chunk_size}_k{k}.json"
    out_path.write_text(json.dumps(run_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] wrote {out_path}")
    print(f"[eval] aggregate: {json.dumps(aggregate)}")
    return run_obj


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in vals if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def _aggregate(results: List[Dict]) -> Dict:
    m = [r["metrics"] for r in results]
    return {
        "n_questions": len(results),
        "mean_faithfulness": _mean([x["faithfulness"] for x in m]),
        "mean_answer_relevance": _mean([x["answer_relevance"] for x in m]),
        "mean_context_precision": _mean([x["context_precision"] for x in m]),
        "refusal_accuracy": _mean([1.0 if x["refusal_correct"] else 0.0 for x in m]),
    }


def main():
    ap = argparse.ArgumentParser(description="Run RAG over the gold set and score it.")
    ap.add_argument("--chunk-size", type=int, default=CONFIG.chunk_size)
    ap.add_argument("--overlap", type=int, default=CONFIG.chunk_overlap)
    ap.add_argument("--k", type=int, default=CONFIG.top_k)
    ap.add_argument("--backend", type=str, default=None, help="groq | ollama")
    args = ap.parse_args()
    run(args.chunk_size, args.overlap, args.k, backend=args.backend)


if __name__ == "__main__":
    main()

"""Evaluation metrics — small, transparent, and documented.

Four signals, deliberately of different kinds so the dashboard tells a real
story rather than three views of one number:

* **faithfulness**       LLM-as-judge: fraction of the answer's factual claims
                         that are supported by the retrieved context. Catches
                         hallucination. Skipped (N/A) for refusals.
* **answer_relevance**   embedding cosine similarity between question and
                         answer. Deterministic, no extra LLM call. Does the
                         answer actually address the question?
* **context_precision**  of the retrieved chunks, the fraction drawn from the
                         gold ``source_docs``. Pure retrieval quality. N/A for
                         the unanswerable question (no gold sources).
* **refusal_correct**    did the model answer vs. refuse correctly? The right
                         behaviour on the unanswerable question is refusal.

The LLM judge calls the same free generation backend, so evaluation is also $0.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

import numpy as np

from .providers import Embedder
from .rag import REFUSAL_TEXT


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def strip_sources_line(answer: str) -> str:
    """Remove the trailing 'Sources: ...' line so it isn't judged as a claim."""
    lines = [ln for ln in answer.splitlines() if not ln.strip().lower().startswith("sources:")]
    return "\n".join(lines).strip()


def is_refusal(answer: str) -> bool:
    """True when the model declined to answer. Matches the canonical refusal
    phrase, tolerant of light punctuation/casing drift from the model."""
    norm = re.sub(r"[^a-z ]", "", answer.lower())
    target = re.sub(r"[^a-z ]", "", REFUSAL_TEXT.lower())
    return target in norm


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def context_precision(contexts: List[Dict], gold_source_docs: List[str]) -> Optional[float]:
    """Fraction of retrieved chunks whose source is in the gold source docs.
    Returns None when there are no gold sources (the unanswerable question)."""
    if not gold_source_docs:
        return None
    if not contexts:
        return 0.0
    gold = set(gold_source_docs)
    hits = sum(1 for c in contexts if c.get("source") in gold)
    return round(hits / len(contexts), 4)


def answer_relevance(question: str, answer: str, embedder: Embedder) -> float:
    """Cosine similarity between question and answer embeddings (0..1)."""
    q = np.array(embedder.embed_query(question))
    a = np.array(embedder.embed_query(answer))
    # vectors are already L2-normalised, so dot product == cosine similarity
    sim = float(np.dot(q, a))
    return round(max(0.0, min(1.0, sim)), 4)


_JUDGE_SYSTEM = (
    "You are a strict grounding evaluator. You will be given CONTEXT passages "
    "and an ANSWER. Identify each distinct factual claim in the ANSWER and "
    "decide whether it is directly supported by the CONTEXT. A claim is "
    "supported ONLY if the context states it; do not use outside knowledge. "
    'Respond with strict JSON only: '
    '{"claims": [{"claim": "...", "supported": true|false}]}'
)


def faithfulness(answer: str, contexts: List[Dict],
                 judge: Callable[..., str]) -> Optional[float]:
    """Fraction of answer claims supported by context, via LLM-as-judge.
    Returns None for a refusal (nothing to ground) or if judging fails."""
    body = strip_sources_line(answer)
    if not body or is_refusal(answer):
        return None

    context_str = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(contexts, 1))
    prompt = (
        f"CONTEXT:\n{context_str}\n\n"
        f"ANSWER:\n{body}\n\n"
        "Return the JSON described in the instructions."
    )
    try:
        raw = judge(prompt, system=_JUDGE_SYSTEM, json_mode=True, temperature=0.0)
        data = _loads_lenient(raw)
        claims = data.get("claims", [])
        if not claims:
            return None
        supported = sum(1 for c in claims if c.get("supported") is True)
        return round(supported / len(claims), 4)
    except Exception:
        return None


def refusal_correct(answer: str, expect_refusal: bool) -> bool:
    """Did the model's answer/refuse decision match what the question warranted?"""
    return is_refusal(answer) == expect_refusal


def _loads_lenient(raw: str) -> Dict:
    """Parse JSON, tolerating models that wrap it in prose or code fences."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON object found in judge output")

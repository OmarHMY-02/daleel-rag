"""Daleel — Evaluation dashboard.

Reads precomputed eval runs from ``eval/eval_runs/*.json``. Needs no API key and
no model: the scores are committed to the repo, so this page loads instantly and
for free. Shows aggregate scorecards, a colour-coded per-question table, and a
config-comparison view — the "I gated deployments against benchmarks" story.
"""

import json
import sys
from pathlib import Path

# Make the repo root importable and locate the eval runs.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

EVAL_RUNS_DIR = ROOT / "eval" / "eval_runs"
METRIC_COLS = ["faithfulness", "answer_relevance", "context_precision"]

st.set_page_config(page_title="Daleel — Evaluation", page_icon="📊", layout="wide")


@st.cache_data
def load_runs():
    runs = []
    for p in sorted(EVAL_RUNS_DIR.glob("*.json")):
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    return runs


def run_label(run) -> str:
    c = run["config"]
    return f"{c.get('label', 'run')} · {c['backend']}/{c['model']} · {run['created_at'][:10]}"


def _color(val):
    """Red→yellow→green background for a 0..1 score; grey for N/A."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "background-color: #eee; color: #888"
    v = max(0.0, min(1.0, float(val)))
    if v < 0.5:
        r, g = 220, int(120 + 200 * (v / 0.5))
    else:
        r, g = int(220 - 180 * ((v - 0.5) / 0.5)), 200
    return f"background-color: rgb({r},{g},120)"


def per_question_table(run):
    rows = []
    for r in run["results"]:
        m = r["metrics"]
        rows.append({
            "id": r["id"],
            "type": r["type"],
            "faithfulness": m["faithfulness"],
            "answer_relevance": m["answer_relevance"],
            "context_precision": m["context_precision"],
            "refusal_ok": "✓" if m["refusal_correct"] else "✗",
        })
    df = pd.DataFrame(rows).set_index("id")
    try:
        styled = df.style.map(_color, subset=METRIC_COLS)
    except AttributeError:  # pandas < 2.1
        styled = df.style.applymap(_color, subset=METRIC_COLS)
    return styled.format("{:.2f}", subset=METRIC_COLS, na_rep="—")


def scorecard(agg):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness", _fmt(agg["mean_faithfulness"]))
    c2.metric("Answer relevance", _fmt(agg["mean_answer_relevance"]))
    c3.metric("Context precision", _fmt(agg["mean_context_precision"]))
    c4.metric("Refusal accuracy", _fmt(agg["refusal_accuracy"]))


def _fmt(v):
    return "—" if v is None else f"{v:.2f}"


def main():
    st.title("📊 Daleel — Evaluation Dashboard")
    st.caption(
        "Faithfulness, answer relevance, and context precision across the gold "
        "question set. Precomputed and committed — no key, no model, $0 to view."
    )

    runs = load_runs()
    if not runs:
        st.info(
            "No eval runs found. Generate some with "
            "`python -m src.run_eval --chunk-size 500 --k 4`."
        )
        return

    with st.expander("ℹ️  How the metrics are defined"):
        st.markdown(_METHODOLOGY)

    labels = [run_label(r) for r in runs]

    # ---- Single-run scorecard + per-question detail --------------------- #
    st.subheader("Run scorecard")
    idx = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i])
    run = runs[idx]
    scorecard(run["aggregate"])
    st.dataframe(per_question_table(run), use_container_width=True)

    st.caption("Expand a question to see ground truth, the generated answer, and retrieved chunks.")
    for r in run["results"]:
        flag = "🟢" if r["metrics"]["refusal_correct"] else "🔴"
        with st.expander(f"{flag} {r['id']} · {r['type']} — {r['question']}"):
            st.markdown(f"**Ground truth:** {r['ground_truth']}")
            st.markdown(f"**Generated answer:** {r['answer']}")
            st.markdown(f"**Gold source docs:** {', '.join(r['gold_source_docs']) or '— (unanswerable)'}")
            st.markdown("**Retrieved chunks:**")
            for c in r["retrieved"]:
                st.markdown(f"- `{c['source']}` — {c['heading']} · sim `{c['score']}`")

    # ---- Comparison ----------------------------------------------------- #
    if len(runs) >= 2:
        st.subheader("Compare configurations")
        col_a, col_b = st.columns(2)
        ia = col_a.selectbox("Baseline", range(len(runs)), index=0,
                             format_func=lambda i: labels[i], key="cmp_a")
        ib = col_b.selectbox("Candidate", range(len(runs)), index=1,
                             format_func=lambda i: labels[i], key="cmp_b")
        _comparison(runs[ia]["aggregate"], runs[ib]["aggregate"])


def _comparison(a, b):
    keys = [
        ("mean_faithfulness", "Faithfulness"),
        ("mean_answer_relevance", "Answer relevance"),
        ("mean_context_precision", "Context precision"),
        ("refusal_accuracy", "Refusal accuracy"),
    ]
    cols = st.columns(len(keys))
    for col, (key, name) in zip(cols, keys):
        av, bv = a.get(key), b.get(key)
        delta = None if (av is None or bv is None) else round(bv - av, 3)
        col.metric(name, _fmt(bv), delta=delta)


_METHODOLOGY = """
- **Faithfulness** — LLM-as-judge: of the factual claims in the generated answer,
  the fraction directly supported by the retrieved context. Catches
  hallucination. *N/A for a refusal (nothing to ground).*
- **Answer relevance** — cosine similarity between the question and answer
  embeddings. Deterministic, no extra LLM call. Does the answer address the
  question asked?
- **Context precision** — of the retrieved chunks, the fraction drawn from the
  question's gold `source_docs`. Pure retrieval quality. *N/A for the
  unanswerable question, which has no gold sources.*
- **Refusal accuracy** — did the model correctly answer vs. refuse? The gold set
  includes one unanswerable question where the *correct* behaviour is refusal.

The gold set is deliberately adversarial — a numeric-reasoning question, two
multi/cross-document questions, and one unanswerable question — so the scores
aren't trivially perfect. The judge calls the same free backend, so evaluation
costs $0.
"""


main()

"""Daleel — RAG chat UI.

Ask a question over the Najmah document corpus and get an answer with citations
to the source documents. Embeddings run locally (free); generation runs on the
configured backend (Groq free tier in the hosted demo, Ollama for local dev).
The visitor needs no key — the host provides one as a secret.
"""

import sys
from pathlib import Path

# Make the repo root importable so `from src import ...` works under Streamlit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.config import CONFIG  # noqa: E402
from src.providers import (  # noqa: E402
    BackendUnavailableError,
    Embedder,
    RateLimitedError,
    backend_ready,
)
from src import rag  # noqa: E402

st.set_page_config(page_title="Daleel — RAG Q&A", page_icon="🧭", layout="centered")

SUGGESTIONS = [
    "How many annual leave days do confirmed full-time employees get?",
    "Can I return opened cosmetics?",
    "What spend reaches Platinum tier, and what are the perks?",
    "How quickly must a data breach be reported, and to whom?",
    "Does Najmah offer a free gym membership?",  # deliberately unanswerable
]


@st.cache_resource(show_spinner="Loading the embedding model (first load only)…")
def _warm_embedder():
    """Load the local embedding model once and reuse across reruns."""
    return Embedder()


def main():
    st.title("🧭 Daleel — Policy Q&A")
    st.caption(
        "Retrieval-augmented answers over Najmah Retail Group's policy documents, "
        "with citations. Runs on free open models — local embeddings + hosted Llama. "
        "No data is stored."
    )

    _warm_embedder()  # warm the model up front so the first answer isn't slow

    # ---- Sidebar: backend status + controls ----------------------------- #
    with st.sidebar:
        st.subheader("Settings")
        ok, status = backend_ready()
        if ok:
            st.success(f"Generation: {status}")
        else:
            st.warning(f"Generation: {status}")

        byok = ""
        if CONFIG.llm_backend == "groq" and not ok:
            st.info(
                "No host key is configured. Paste your own free Groq key to try "
                "the live demo (kept only in this session)."
            )
            byok = st.text_input("Groq API key", type="password")

        k = st.slider("Passages retrieved (k)", 2, 6, CONFIG.top_k)

        st.divider()
        st.caption("**Try a question:**")
        for s in SUGGESTIONS:
            if st.button(s, width="stretch"):
                st.session_state["pending"] = s
                st.rerun()

        st.divider()
        st.caption(
            "Corpus: 6 fictional UAE retail policy docs. The embedding model is "
            "multilingual, so an Arabic corpus drops in without pipeline changes."
        )

    # ---- Conversation --------------------------------------------------- #
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("contexts"):
                _render_citations(msg["contexts"])

    pending = st.session_state.pop("pending", None)
    typed = st.chat_input("Ask about leave, returns, loyalty, benefits, data…")
    question = typed or pending
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                result = rag.answer(question, k=k, api_key=(byok or None))
            except RateLimitedError:
                st.warning(
                    "The free generation backend is rate-limited right now. "
                    "Please wait a few seconds and try again."
                )
                return
            except BackendUnavailableError as e:
                st.error(str(e))
                return
            except Exception as e:  # pragma: no cover - defensive UI guard
                st.error(f"Something went wrong: {e}")
                return

        st.markdown(result["answer"])
        _render_citations(result["contexts"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "contexts": result["contexts"]}
    )


def _render_citations(contexts):
    if not contexts:
        return
    srcs = ", ".join(sorted({c["source"] for c in contexts}))
    with st.expander(f"📎 Sources & retrieved passages ({srcs})"):
        for i, c in enumerate(contexts, 1):
            st.markdown(
                f"**[{i}] {c['source']} — {c['heading']}**  ·  similarity `{c['score']}`"
            )
            st.caption(c["text"])


main()

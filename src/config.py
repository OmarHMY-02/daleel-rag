"""Central configuration for Daleel.

Everything that the pipeline needs to know is read from environment variables
(with sane defaults) in one place, so the free-hosted path (Groq) and the
fully-offline path (Ollama) are switchable by config alone. A local ``.env`` is
loaded if present; on Streamlit Cloud the same keys come from host secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load a local .env during development
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass

# Repository root (this file lives in <root>/src/config.py).
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
EVAL_DIR = ROOT / "eval"
EVAL_RUNS_DIR = EVAL_DIR / "eval_runs"
GOLD_QA_PATH = EVAL_DIR / "gold_qa.json"


def _get(name: str, default: str) -> str:
    """Read an env var, also honouring Streamlit secrets when available."""
    val = os.environ.get(name)
    if val:
        return val
    try:  # Streamlit secrets are only present inside a running app
        import streamlit as st

        if name in st.secrets:  # type: ignore[attr-defined]
            return str(st.secrets[name])
    except Exception:
        pass
    return default


@dataclass(frozen=True)
class Config:
    # Generation backend selection
    llm_backend: str = _get("LLM_BACKEND", "groq").lower()

    # Groq (free hosted)
    groq_api_key: str = _get("GROQ_API_KEY", "")
    groq_model: str = _get("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Ollama (local offline)
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.1:8b")
    ollama_host: str = _get("OLLAMA_HOST", "http://localhost:11434")

    # Embeddings + vector store
    embed_model: str = _get("EMBED_MODEL", "intfloat/multilingual-e5-small")
    chroma_dir: str = _get("CHROMA_DIR", str(ROOT / "chroma"))

    # Retrieval / chunking defaults
    top_k: int = int(_get("TOP_K", "4"))
    chunk_size: int = int(_get("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(_get("CHUNK_OVERLAP", "75"))


CONFIG = Config()


def collection_name(chunk_size: int, chunk_overlap: int) -> str:
    """Each chunking configuration gets its own Chroma collection so the eval
    can compare configs (e.g. chunk 300 vs 500) against a matching index."""
    return f"najmah_c{chunk_size}_o{chunk_overlap}"

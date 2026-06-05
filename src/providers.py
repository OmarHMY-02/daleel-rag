"""Model access behind one thin interface.

Two things the pipeline needs a model for:

* **Embeddings** — always local, via ``sentence-transformers`` (no key, no cost,
  runs on the free CPU host). The model is an E5 model, which requires the
  ``query:`` / ``passage:`` prefixes — getting those right is what makes
  retrieval actually work, so they live here and nowhere else.
* **Generation** — swappable backend selected by ``LLM_BACKEND``:
  ``groq`` (free hosted Llama, used in the live demo) or ``ollama`` (local,
  offline dev). Both are reached through one ``generate()`` call so the rest of
  the codebase never knows or cares which is running.
"""

from __future__ import annotations

import functools
from typing import List, Optional

from .config import CONFIG


# --------------------------------------------------------------------------- #
# Embeddings (local sentence-transformers, E5 with required prefixes)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=2)
def _load_embedder(model_name: str):
    # Imported lazily so that importing this module (e.g. the eval dashboard,
    # which needs no model at all) stays cheap.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    """E5 embeddings with the mandatory ``query:`` / ``passage:`` prefixes.

    E5 was trained with these prefixes; omitting them silently degrades
    retrieval. Embeddings are L2-normalised so a dot product equals cosine
    similarity, matching the cosine space configured on the Chroma collection.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or CONFIG.embed_model
        self.model = _load_embedder(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            inputs, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(
            [f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec[0].tolist()

    def token_len(self, text: str) -> int:
        """Token count under the embedding model's own tokenizer — used by the
        chunker so chunks respect the model's 512-token context window."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# Generation (Groq | Ollama behind one interface)
# --------------------------------------------------------------------------- #
class RateLimitedError(RuntimeError):
    """Raised when the generation backend is rate-limited, so the UI can show a
    friendly 'try again in a moment' message instead of a stack trace."""


class BackendUnavailableError(RuntimeError):
    """Raised when the selected backend is not usable (missing key, server down)."""


def generate(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    backend: Optional[str] = None,
    api_key: Optional[str] = None,
    json_mode: bool = False,
) -> str:
    """Generate a completion from the configured backend.

    ``backend`` / ``api_key`` may be overridden per-call (the live app passes a
    visitor-supplied key as a fallback when the host key is rate-limited).
    ``json_mode`` asks the backend for strict JSON — used by the LLM-as-judge
    metrics so their output parses reliably.
    """
    backend = (backend or CONFIG.llm_backend).lower()
    if backend == "groq":
        return _generate_groq(prompt, system, temperature, max_tokens, api_key, json_mode)
    if backend == "ollama":
        return _generate_ollama(prompt, system, temperature, max_tokens, json_mode)
    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (expected 'groq' or 'ollama')")


def _generate_groq(prompt, system, temperature, max_tokens, api_key, json_mode) -> str:
    from groq import Groq
    from groq import RateLimitError as GroqRateLimitError

    key = api_key or CONFIG.groq_api_key
    if not key:
        raise BackendUnavailableError(
            "GROQ_API_KEY is not set. Add it to your .env locally, or to the host "
            "secrets on deploy. (Or set LLM_BACKEND=ollama to run fully offline.)"
        )

    client = Groq(api_key=key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=CONFIG.groq_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except GroqRateLimitError as e:  # surface a typed error for the UI
        raise RateLimitedError(str(e)) from e
    return resp.choices[0].message.content or ""


def _generate_ollama(prompt, system, temperature, max_tokens, json_mode) -> str:
    import requests

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": CONFIG.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        r = requests.post(f"{CONFIG.ollama_host}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
    except requests.RequestException as e:
        raise BackendUnavailableError(
            f"Could not reach Ollama at {CONFIG.ollama_host}. Is `ollama serve` "
            f"running and is the model '{CONFIG.ollama_model}' pulled? ({e})"
        ) from e
    return r.json().get("message", {}).get("content", "")


def backend_ready(backend: Optional[str] = None) -> tuple[bool, str]:
    """Lightweight readiness check for the UI status indicator. Returns
    ``(ok, human_message)`` without making an expensive generation call."""
    backend = (backend or CONFIG.llm_backend).lower()
    if backend == "groq":
        if CONFIG.groq_api_key:
            return True, f"Groq · {CONFIG.groq_model}"
        return False, "Groq key not configured"
    if backend == "ollama":
        return True, f"Ollama · {CONFIG.ollama_model}"
    return False, f"Unknown backend {backend!r}"

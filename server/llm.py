"""
LLM inference module.
Supports two backends:
  - vllm  (preferred on JarvisLabs — fastest, OpenAI-compatible API)
  - ollama (easier local setup, slightly slower)

Set BACKEND = "vllm" or "ollama" below.
"""

import os
from typing import Iterator

import httpx

# ── Config ──────────────────────────────────────────────────────────────────
BACKEND = os.getenv("LLM_BACKEND", "ollama")   # "vllm" | "ollama"
MODEL   = os.getenv("LLM_MODEL", "mistral")    # "mistral" | "llama3" | etc.

VLLM_BASE_URL   = os.getenv("VLLM_URL",   "http://localhost:8000/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. "
    "Keep responses short — 1 to 3 sentences max — because they will be spoken aloud. "
    "Avoid markdown, lists, or special characters."
)

# Conversation memory (grows per session)
_history: list[dict] = []


def reset_history():
    """Clear conversation memory (call at session start)."""
    global _history
    _history = []


def chat(user_message: str) -> str:
    """
    Send a user message, get full LLM response string.
    Updates in-memory conversation history.
    """
    _history.append({"role": "user", "content": user_message})

    if BACKEND == "vllm":
        reply = _vllm_chat(_history)
    else:
        reply = _ollama_chat(_history)

    _history.append({"role": "assistant", "content": reply})
    print(f"[LLM] Response: {reply}")
    return reply


def stream_chat(user_message: str) -> Iterator[str]:
    """
    Stream LLM tokens as they arrive.
    Yields text chunks; collects full reply for history at the end.
    """
    _history.append({"role": "user", "content": user_message})

    chunks = []
    if BACKEND == "vllm":
        gen = _vllm_stream(_history)
    else:
        gen = _ollama_stream(_history)

    for chunk in gen:
        chunks.append(chunk)
        yield chunk

    full_reply = "".join(chunks)
    _history.append({"role": "assistant", "content": full_reply})
    print(f"[LLM] Streamed: {full_reply}")


# ── vllm backend ─────────────────────────────────────────────────────────────
def _vllm_chat(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_tokens": 256,
        "temperature": 0.7,
    }
    r = httpx.post(f"{VLLM_BASE_URL}/chat/completions", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _vllm_stream(messages: list[dict]) -> Iterator[str]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": True,
    }
    with httpx.stream("POST", f"{VLLM_BASE_URL}/chat/completions",
                      json=payload, timeout=60) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                import json
                data = json.loads(line[6:])
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta


# ── ollama backend ────────────────────────────────────────────────────────────
def _ollama_chat(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": False,
    }
    r = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _ollama_stream(messages: list[dict]) -> Iterator[str]:
    import json as _json
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": True,
    }
    with httpx.stream("POST", f"{OLLAMA_BASE_URL}/api/chat",
                      json=payload, timeout=60) as r:
        for line in r.iter_lines():
            if line:
                data = _json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing LLM module (backend:", BACKEND, "model:", MODEL, ")")
    reply = chat("What's the capital of France? One sentence only.")
    print("Reply:", reply)

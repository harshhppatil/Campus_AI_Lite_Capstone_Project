"""
config.py — LLM client setup and environment loading for CampusAI Lite.

All agent and LLM code imports from here so that swapping providers is a
one-line change in .env (LLM_MODEL=<model_name>).
"""

import os
import time
import functools
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ── API keys & model config ────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Return a LangChain-wrapped Gemini LLM instance."""
    key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    if not key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=key,
        temperature=temperature,
    )


# ── Rate-limit retry decorator ─────────────────────────────────────────────
def with_retry(max_attempts: int = 3, delay_seconds: float = 5.0):
    """
    Simple retry-with-backoff decorator for LLM calls.
    Handles Gemini free-tier rate limit errors (429 / ResourceExhausted).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc).lower()
                    # Retry only on rate-limit or transient errors
                    if any(k in err_str for k in ("429", "rate", "quota", "resource exhausted", "503", "500")):
                        wait = delay_seconds * attempt
                        if VERBOSE:
                            print(f"[retry] attempt {attempt}/{max_attempts} failed ({exc}); waiting {wait}s")
                        time.sleep(wait)
                    else:
                        raise
            raise last_exc
        return wrapper
    return decorator

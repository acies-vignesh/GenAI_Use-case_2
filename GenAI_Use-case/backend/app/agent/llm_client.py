"""Thin provider-agnostic LLM wrapper (structured JSON output, retries, token accounting).

Any OpenAI-compatible endpoint works (Groq, Ollama, vLLM, OpenAI...): only LLM_BASE_URL / LLM_MODEL change.

- JSON mode: the model is told to answer with a JSON object only.
- Structured output: the JSON is validated with a Pydantic model; on failure the validation errors
  are sent back once so the model can correct itself.
- Cache: responses are stored under data/llm_cache keyed by a hash of (model, temperature, messages).
  Same input -> same output, no cost, and it eases free-tier rate limits.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import PROJECT_ROOT, Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache"


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Base class: subclasses implement `chat`; structured output + retry live here."""
    model_name = "unknown"

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def complete_structured(self, system: str, user: str, model_cls: type[T], retries: int = 1,
                            prior: list[dict] | None = None) -> T:
        """`prior`: earlier turns of the conversation (e.g. the first answer, before a repair request)."""
        messages = [{"role": "system", "content": system}, *(prior or []), {"role": "user", "content": user}]
        for attempt in range(retries + 1):
            text = self.chat(messages)
            try:
                return model_cls.model_validate_json(_strip_fences(text))
            except ValidationError as e:
                problem = e.errors(include_url=False)
            except ValueError as e:  # not JSON at all
                problem = str(e)
            log.warning("LLM output invalid (attempt %d): %s", attempt + 1, str(problem)[:500])
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Your JSON did not match the required structure: {problem}. "
                                            "Return the corrected JSON object only."},
            ]
        raise LLMError(f"LLM output still invalid after {retries + 1} attempts")


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences even in JSON mode."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()


class OpenAICompatibleClient(LLMClient):
    def __init__(self, settings: Settings, cache_dir: Path | None = DEFAULT_CACHE_DIR, max_tokens: int = 8000):
        if not settings.llm_api_key:
            raise LLMError("LLM_API_KEY is not set in .env")
        from openai import OpenAI   # imported here so --no-llm runs don't need the package configured

        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                              max_retries=3, timeout=120)
        self.model_name = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages: list[dict]) -> str:
        key = hashlib.sha256(json.dumps([self.model_name, self.temperature, messages]).encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            self.usage["cached"] += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))["response"]

        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        self.usage["calls"] += 1
        if resp.usage:
            self.usage["prompt_tokens"] += resp.usage.prompt_tokens
            self.usage["completion_tokens"] += resp.usage.completion_tokens

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"model": self.model_name, "messages": messages, "response": text},
                                             indent=1), encoding="utf-8")
        return text

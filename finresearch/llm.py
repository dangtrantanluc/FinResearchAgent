"""The only place that talks to a language model.

Everything the agent asks of the model is "fill this Pydantic schema", so the
provider is a one-class choice. Model names come from the environment because
free-tier availability changes; see .env.example.
"""
from __future__ import annotations

import os
import time
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
Tier = Literal["fast", "strong"]


class LLM(Protocol):
    def generate(self, schema: type[T], system: str, prompt: str, tier: Tier = "fast") -> T: ...


class GeminiLLM:
    """Gemini through the google-genai SDK.

    Each tier is an ordered list of models; the first one that answers wins. That is
    what makes a free-tier key usable. Observed on 2026-10-08: the free tier allows 20
    requests per day per model, the Pro models have no free quota at all (429), and the
    newest Flash models are overloaded (503) for minutes at a time. A report costs two
    requests, one per tier, so the two lists share no model until their last resort.
    """

    DEFAULTS = {"fast": "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash",
                "strong": "gemini-3.5-flash,gemini-3-flash-preview,gemini-2.5-flash"}
    ROUND_WAITS = (0, 8, 25)  # seconds before each pass through the list; a 503 usually clears within half a minute
    TIMEOUT_MS = 180_000

    def __init__(self, api_key: str | None = None):
        from google import genai
        from google.genai import types

        # The SDK's own retries back off for minutes on 429/503 and hide which model finally answered.
        options = types.HttpOptions(timeout=self.TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1))
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"], http_options=options)
        self.models = {tier: [m.strip() for m in os.environ.get(f"GEMINI_MODEL_{tier.upper()}", default).split(",") if m.strip()]
                       for tier, default in self.DEFAULTS.items()}
        self.calls: list[dict] = []  # one entry per attempt: model, tier, seconds, and tokens or the error

    def _call(self, model: str, schema: type[T], system: str, prompt: str):
        return self.client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "system_instruction": system,
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_json_schema": schema.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )

    def _request(self, tier: Tier, schema: type[T], system: str, prompt: str):
        """One answer from the first model of the tier that responds."""
        from google.genai import errors

        last: Exception | None = None
        failures: dict[str, int] = {}
        for wait in self.ROUND_WAITS:
            time.sleep(wait)
            for model in self.models[tier]:
                started = time.time()
                record = {"model": model, "tier": tier, "schema": schema.__name__}
                try:
                    response = self._call(model, schema, system, prompt)
                except errors.APIError as exc:
                    last = exc
                    failures[model] = exc.code
                    self.calls.append({**record, "seconds": round(time.time() - started, 1), "error": f"{exc.code} {str(exc.message)[:120]}"})
                    continue
                usage = response.usage_metadata
                self.calls.append({**record, "seconds": round(time.time() - started, 1),
                                   "prompt_tokens": usage.prompt_token_count, "output_tokens": usage.candidates_token_count})
                return response
        tried = ", ".join(f"{m}: {code}" for m, code in failures.items())
        raise RuntimeError(f"Gemini không trả lời được ({tried}). 429 = hết hạn mức, 503 = quá tải, 404 = model đã ngừng.") from last

    def generate(self, schema: type[T], system: str, prompt: str, tier: Tier = "fast") -> T:
        error = None
        for _ in range(2):  # one retry, telling the model what was wrong with its JSON
            ask = prompt if error is None else f"{prompt}\n\nYour previous answer did not fit the schema:\n{error}\nAnswer again."
            response = self._request(tier, schema, system, ask)
            try:
                return schema.model_validate_json(response.text or "")
            except ValidationError as exc:
                error = str(exc)[:1500]
        raise ValueError(f"{self.calls[-1]['model']} did not return valid {schema.__name__}: {error}")


def default_llm() -> LLM | None:
    """Gemini when a key is configured; None means the agent runs in its offline, template-only mode."""
    return GeminiLLM() if os.environ.get("GEMINI_API_KEY") else None


if __name__ == "__main__":  # smoke test: python -m finresearch.llm
    from .env import load_env

    class Reply(BaseModel):
        language: str
        greeting: str

    load_env()
    llm = default_llm()
    if llm is None:
        raise SystemExit("GEMINI_API_KEY chưa được đặt trong .env")
    for tier in ("fast", "strong"):
        reply = llm.generate(Reply, "Answer in Vietnamese.", "Say hello in one short sentence.", tier)
        print(f"{tier}: thứ tự thử {' → '.join(llm.models[tier])}; trả lời bởi {llm.calls[-1]['model']} trong {llm.calls[-1]['seconds']}s: {reply.greeting}")
    for call in llm.calls:
        if "error" in call:
            print(f"  lỗi đã bỏ qua: {call['model']}: {call['error']}")

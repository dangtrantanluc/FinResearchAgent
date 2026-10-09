from types import SimpleNamespace

import pytest
from google.genai import errors
from pydantic import BaseModel

from finresearch.llm import GeminiLLM


class Reply(BaseModel):
    answer: str


def _response(text):
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5))


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL_STRONG", "new-model, old-model")
    monkeypatch.setenv("GEMINI_MODEL_FAST", "old-model")
    monkeypatch.setattr(GeminiLLM, "ROUND_WAITS", (0, 0))
    return GeminiLLM(api_key="not-a-real-key")


def test_an_overloaded_model_falls_back_to_the_next_in_the_list(llm, monkeypatch):
    def call(model, schema, system, prompt):
        if model == "new-model":
            raise errors.APIError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
        return _response('{"answer": "ok"}')

    monkeypatch.setattr(llm, "_call", call)
    assert llm.generate(Reply, "sys", "prompt", tier="strong").answer == "ok"
    assert [(c["model"], "error" in c) for c in llm.calls] == [("new-model", True), ("old-model", False)]


def test_invalid_json_is_sent_back_once_with_the_error(llm, monkeypatch):
    prompts = []

    def call(model, schema, system, prompt):
        prompts.append(prompt)
        return _response('{"wrong": 1}' if len(prompts) == 1 else '{"answer": "fixed"}')

    monkeypatch.setattr(llm, "_call", call)
    assert llm.generate(Reply, "sys", "prompt").answer == "fixed"
    assert len(prompts) == 2 and "did not fit the schema" in prompts[1]


def test_every_model_failing_raises_with_the_list_tried(llm, monkeypatch):
    def call(model, schema, system, prompt):
        raise errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(llm, "_call", call)
    with pytest.raises(RuntimeError, match="new-model: 429, old-model: 429"):
        llm.generate(Reply, "sys", "prompt", tier="strong")
    assert len(llm.calls) == 4  # two passes over two models

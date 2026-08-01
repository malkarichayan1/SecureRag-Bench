import json
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from secure_rag_bench.evaluation.model_adapters import (
    ClaudeAdapter,
    GenerationRequest,
    OpenAICompatibleAdapter,
    TransformersAdapter,
    choose_load_plan,
)


def request() -> GenerationRequest:
    return GenerationRequest("system instruction", "user request", max_new_tokens=37)


def test_generation_request_defaults_to_bounded_generation() -> None:
    assert GenerationRequest("system", "user").max_new_tokens == 512


def test_preflight_skips_32b_when_estimate_exceeds_vram() -> None:
    plan = choose_load_plan(
        {"parameters_b": 32.5, "quantization": "4bit"}, available_vram_gb=16.0
    )

    assert plan.status == "skipped"
    assert plan.reason == "insufficient_vram"
    assert plan.required_vram_gb == pytest.approx(20.5)


def test_preflight_marks_model_ready_when_estimate_fits_vram() -> None:
    plan = choose_load_plan(
        {"parameters_b": 7.0, "quantization": "fp16"}, available_vram_gb=18.0
    )

    assert plan.status == "ready"
    assert plan.reason == ""
    assert plan.required_vram_gb == pytest.approx(17.8)


def test_endpoint_adapter_returns_request_metadata() -> None:
    observed: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers["authorization"]
        observed["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "req_123"},
            json={
                "model": "provider-pinned-70b-20260801",
                "choices": [{"message": {"content": "model output"}}],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handle))
    result = OpenAICompatibleAdapter(
        "optional-70b", "https://example.test/v1", "token", client=client
    ).generate(request())

    assert result.text == "model output"
    assert result.metadata["provider_request_id"] == "req_123"
    assert result.metadata["model_id"] == "optional-70b"
    assert result.metadata["requested_model_id"] == "optional-70b"
    assert result.metadata["provider_model_id"] == "provider-pinned-70b-20260801"
    assert observed == {
        "url": "https://example.test/v1/chat/completions",
        "authorization": "Bearer token",
        "body": {
            "model": "optional-70b",
            "messages": [
                {"role": "system", "content": "system instruction"},
                {"role": "user", "content": "user request"},
            ],
            "max_tokens": 37,
            "temperature": 0,
        },
    }


def test_endpoint_adapter_redacts_authorization_from_errors() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(401, text="Authorization: Bearer secret-token")
        )
    )
    adapter = OpenAICompatibleAdapter(
        "optional-70b", "https://example.test/v1", "secret-token", client=client
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.generate(request())

    assert "secret-token" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_endpoint_adapter_error_traceback_has_no_credential_cause() -> None:
    secret = "openai-live-secret"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.RequestError(f"Authorization: Bearer {secret}", request=request)

    adapter = OpenAICompatibleAdapter(
        "optional-70b",
        "https://example.test/v1",
        secret,
        client=httpx.Client(transport=httpx.MockTransport(fail)),
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.generate(request())

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert secret not in formatted
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_claude_adapter_uses_messages_endpoint_without_sdk() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://anthropic.example/v1/messages"
        assert request.headers["x-api-key"] == "claude-token"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert json.loads(request.content) == {
            "model": "optional-claude",
            "system": "system instruction",
            "messages": [{"role": "user", "content": "user request"}],
            "max_tokens": 37,
            "temperature": 0,
        }
        return httpx.Response(
            200,
            headers={"request-id": "claude_req_456"},
            json={
                "model": "claude-3-5-sonnet-20260801",
                "content": [{"type": "text", "text": "claude output"}],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handle))
    result = ClaudeAdapter(
        "optional-claude", "claude-token", base_url="https://anthropic.example", client=client
    ).generate(request())

    assert result.text == "claude output"
    assert result.metadata["provider_request_id"] == "claude_req_456"
    assert result.metadata["requested_model_id"] == "optional-claude"
    assert result.metadata["provider_model_id"] == "claude-3-5-sonnet-20260801"


def test_claude_adapter_error_traceback_has_no_credential_cause() -> None:
    secret = "claude-live-secret"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.RequestError(f"x-api-key: {secret}", request=request)

    adapter = ClaudeAdapter(
        "optional-claude",
        secret,
        base_url="https://anthropic.example",
        client=httpx.Client(transport=httpx.MockTransport(fail)),
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.generate(request())

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert secret not in formatted
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_transformers_adapter_records_greedy_generation_metadata(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeInputIds(list):
        shape = (1, 2)

    class FakeInputs(dict):
        def to(self, device: str):
            calls["input_device"] = device
            return self

    class FakeTokenizer:
        init_kwargs = {"_commit_hash": "tokenizer-revision"}

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            calls["messages"] = messages
            calls["chat_template"] = (tokenize, add_generation_prompt)
            return "rendered prompt"

        def __call__(self, prompt, *, return_tensors):
            calls["tokenized"] = (prompt, return_tensors)
            return FakeInputs(input_ids=FakeInputIds([1, 2]))

        def decode(self, tokens, *, skip_special_tokens):
            calls["decoded_tokens"] = tokens
            assert skip_special_tokens
            return "generated text"

    class FakeModel:
        device = "cuda:0"
        config = SimpleNamespace(_commit_hash="model-revision")
        hf_device_map = {"model": "cuda:0", "lm_head": "cpu"}

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return [[1, 2, 3, 4]]

    tokenizer = FakeTokenizer()
    model = FakeModel()

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer_load"] = (model_id, kwargs)
            return tokenizer

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_load"] = (model_id, kwargs)
            return model

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            calls["quantization_config"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float16="FLOAT16", no_grad=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeModelFactory,
            AutoTokenizer=FakeTokenizerFactory,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
        ),
    )

    result = TransformersAdapter(
        "local-model", revision="pinned", dtype="float16", quantization="4bit"
    ).generate(request())

    assert result.text == "generated text"
    assert result.metadata == {
        "provider": "transformers",
        "model_id": "local-model",
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "dtype": "float16",
        "quantization": "4bit",
        "generation_args": {"max_new_tokens": 37, "do_sample": False},
        "device_names": ["cuda:0", "cpu"],
    }
    assert calls["chat_template"] == (False, True)
    model_id, model_load_kwargs = calls["model_load"]
    assert model_id == "local-model"
    assert model_load_kwargs["revision"] == "pinned"
    assert model_load_kwargs["torch_dtype"] == "FLOAT16"
    assert model_load_kwargs["device_map"] == "auto"
    assert calls["quantization_config"] == {"load_in_4bit": True}
    assert calls["generate"] == {
        "input_ids": [1, 2],
        "max_new_tokens": 37,
        "do_sample": False,
    }


def test_native_model_config_marks_hosted_entries_disabled() -> None:
    config_path = Path("configs/native_study_models.json")
    models = json.loads(config_path.read_text(encoding="utf-8"))["models"]
    by_name = {model["name"]: model for model in models}

    assert {"qwen-7b", "qwen-14b", "llama-3.1-8b"} <= set(by_name)
    assert by_name["qwen-32b-4bit"]["tier"] == "conditional"
    assert by_name["llama-3.1-70b-endpoint"]["enabled"] is False
    assert by_name["claude"]["enabled"] is False

"""Model-independent text generation adapters for native benchmark studies."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import httpx


@dataclass(frozen=True)
class GenerationRequest:
    """The prompts and deterministic token limit for one generation."""

    system_prompt: str
    user_prompt: str
    max_new_tokens: int = 512


@dataclass(frozen=True)
class GenerationResult:
    """Raw generated text with enough metadata to reproduce its provenance."""

    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadPlan:
    """A conservative local-model memory preflight result."""

    status: str
    required_vram_gb: float
    reason: str


class ModelAdapter(Protocol):
    """A model backend that can generate one benchmark response."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate deterministic text for ``request``."""


def choose_load_plan(
    model_config: Mapping[str, Any], available_vram_gb: float
) -> LoadPlan:
    """Estimate whether a local configuration fits its available VRAM."""
    bits_per_weight = 4 if model_config["quantization"] == "4bit" else 16
    weight_gb = float(model_config["parameters_b"]) * bits_per_weight / 8
    required_vram_gb = weight_gb * 1.20 + 1.0
    if required_vram_gb <= available_vram_gb:
        return LoadPlan("ready", required_vram_gb, "")
    return LoadPlan("skipped", required_vram_gb, "insufficient_vram")


class TransformersAdapter:
    """Lazy local Transformers adapter with deterministic greedy decoding."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        dtype: str = "float16",
        quantization: str = "none",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.quantization = quantization
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        tokenizer, model = self._load()
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        import torch

        generation_args = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": False,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args)
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return GenerationResult(
            text=tokenizer.decode(new_tokens, skip_special_tokens=True),
            metadata={
                "provider": "transformers",
                "model_id": self.model_id,
                "model_revision": _resolved_revision(model, self.revision),
                "tokenizer_revision": _resolved_revision(tokenizer, self.revision),
                "dtype": self.dtype,
                "quantization": self.quantization,
                "generation_args": generation_args,
                "device_names": _device_names(model),
            },
        )

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs: dict[str, Any] = {
            "torch_dtype": _torch_dtype(torch, self.dtype),
            "device_map": "auto",
        }
        if self.revision is not None:
            load_kwargs["revision"] = self.revision
        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.quantization not in {"none", "fp16", "bf16"}:
            raise ValueError(f"unsupported quantization: {self.quantization}")

        tokenizer_kwargs: dict[str, Any] = {}
        if self.revision is not None:
            tokenizer_kwargs["revision"] = self.revision
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        return self._tokenizer, self._model


class OpenAICompatibleAdapter:
    """Optional OpenAI-compatible endpoint adapter using only ``httpx``."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        client: httpx.Client | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"missing endpoint credential: {api_key_env}")
        self._client = client or httpx.Client()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "max_tokens": request.max_new_tokens,
            "temperature": 0,
        }
        response = self._post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
        )
        data = _response_json(response, "OpenAI-compatible")
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI-compatible response is missing generated text") from exc
        if not isinstance(text, str):
            raise RuntimeError("OpenAI-compatible response generated non-text content")
        return GenerationResult(
            text=text,
            metadata={
                "provider": "openai_compatible",
                "model_id": self.model_id,
                "endpoint": self.base_url,
                "provider_request_id": response.headers.get("x-request-id") or data.get("id"),
                "generation_args": {"max_tokens": request.max_new_tokens, "temperature": 0},
            },
        )

    def _post(
        self, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> httpx.Response:
        try:
            response = self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            body = ""
            if isinstance(exc, httpx.HTTPStatusError):
                body = exc.response.text
            raise RuntimeError(
                _redact_error(f"endpoint request failed: {exc}; {body}", self.api_key)
            ) from exc


class ClaudeAdapter:
    """Optional Claude endpoint adapter implemented through ``httpx``."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.anthropic.com",
        api_key_env: str = "ANTHROPIC_API_KEY",
        client: httpx.Client | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"missing endpoint credential: {api_key_env}")
        self._client = client or httpx.Client()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {
            "model": self.model_id,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
            "max_tokens": request.max_new_tokens,
            "temperature": 0,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        try:
            response = self._client.post(
                f"{self.base_url}/v1/messages", headers=headers, json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            body = exc.response.text if isinstance(exc, httpx.HTTPStatusError) else ""
            raise RuntimeError(
                _redact_error(f"Claude request failed: {exc}; {body}", self.api_key)
            ) from exc
        data = _response_json(response, "Claude")
        content = data.get("content") if isinstance(data, dict) else None
        text_parts = [
            block.get("text")
            for block in content or []
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        if not text_parts:
            raise RuntimeError("Claude response is missing generated text")
        return GenerationResult(
            text="".join(text_parts),
            metadata={
                "provider": "claude",
                "model_id": self.model_id,
                "endpoint": self.base_url,
                "provider_request_id": response.headers.get("request-id") or data.get("id"),
                "generation_args": {"max_tokens": request.max_new_tokens, "temperature": 0},
            },
        )


def _torch_dtype(torch: Any, dtype: str) -> Any:
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype: {dtype}") from exc


def _resolved_revision(component: Any, fallback: str | None) -> str | None:
    config = getattr(component, "config", None)
    init_kwargs = getattr(component, "init_kwargs", None)
    for source in (config, init_kwargs):
        if isinstance(source, Mapping) and source.get("_commit_hash"):
            return str(source["_commit_hash"])
        revision = getattr(source, "_commit_hash", None)
        if revision:
            return str(revision)
    return fallback


def _device_names(model: Any) -> list[str]:
    devices = getattr(model, "hf_device_map", {}).values()
    names = [str(device) for device in devices]
    if not names and getattr(model, "device", None) is not None:
        names.append(str(model.device))
    return list(dict.fromkeys(names))


def _response_json(response: httpx.Response, provider: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{provider} response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{provider} response JSON must be an object")
    return data


def _redact_error(message: str, api_key: str) -> str:
    redacted = message.replace(api_key, "[REDACTED]")
    return re.sub(
        r"(?i)(authorization\s*:\s*)(?:bearer\s+)?[^\s,;]+",
        r"\1[REDACTED]",
        redacted,
    )

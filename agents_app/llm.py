"""Model routing and the offline mock LLM.

Provider order is resolved in ``config.Settings.resolved_provider``: Azure, then
OpenAI (or any OpenAI-compatible local server), then an in-process Hugging Face
model, then ``mock``. The mock is a first-class provider: agents pass a
deterministic ``mock=`` value, so offline runs exercise the real workflow with
zero keys and zero network.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from agents_app.config import Settings
from agents_app.config import settings as default_settings
from agents_app.monitoring import RequestMetrics
from agents_app.resilience import CircuitBreaker, RequestBudget


class LLMError(Exception):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text).split()) * 4 // 3)


class LLM:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.provider = self.settings.resolved_provider
        self.breaker = CircuitBreaker()
        self._local_pipelines: dict[str, Any] = {}
        self.mock_mode = os.environ.get("MOCK_LLM_MODE", "normal").lower()

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def describe(self) -> dict:
        return {"provider": self.provider, "mock_mode": self.mock_mode, "breaker": self.breaker.state}

    # ------------------------------------------------------------------ public
    async def complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.0,
        fast: bool = False,
        budget: RequestBudget | None = None,
        metrics: RequestMetrics | None = None,
        mock: Any = None,
    ) -> LLMResult:
        """One completion. ``mock`` is returned verbatim in offline mode."""
        started = time.perf_counter()
        if budget:
            budget.before_call()

        if self.is_mock:
            text = self._mock_text(mock)
            result = self._result(text, "mock", started, system + user, text)
        else:
            result = await self._call_provider(system, user, schema, temperature, fast, started)

        if budget:
            budget.after_call(result.cost_usd)
        if metrics:
            metrics.add_llm(
                result.model, result.latency_s, result.tokens_in, result.tokens_out, result.cost_usd
            )
        return result

    # ------------------------------------------------------------------ mock
    def _mock_text(self, mock: Any) -> str:
        if self.mock_mode == "fail":
            raise LLMError("mock provider outage 503")
        if mock is None:
            return "[mock LLM]"
        if isinstance(mock, str):
            return mock
        return json.dumps(mock)

    # ------------------------------------------------------------ real providers
    async def _call_provider(self, system, user, schema, temperature, fast, started) -> LLMResult:
        prompt_system = system
        if schema and self.settings.json_mode:
            prompt_system += (
                "\nReturn ONLY a JSON object that validates against this schema:\n"
                + json.dumps(schema)
            )

        async def _invoke() -> str:
            if self.provider == "azure":
                return await asyncio.to_thread(
                    self._call_azure, prompt_system, user, temperature, fast
                )
            if self.provider == "openai":
                return await asyncio.to_thread(
                    self._call_openai, prompt_system, user, temperature, fast
                )
            if self.provider == "local":
                return await asyncio.to_thread(
                    self._call_local, prompt_system, user, temperature, fast
                )
            raise LLMError(f"unsupported LLM provider: {self.provider}")

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                text = await asyncio.wait_for(
                    self.breaker.call(_invoke), timeout=self.settings.llm_timeout_s
                )
                model = self._model_name(fast)
                return self._result(text, model, started, prompt_system + user, text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.05)
        raise LLMError(f"LLM call failed: {last_error}")

    def _model_name(self, fast: bool) -> str:
        if self.provider == "azure":
            return (
                self.settings.azure_openai_deployment_fast
                if fast and self.settings.azure_openai_deployment_fast
                else self.settings.azure_openai_deployment
            ) or "azure-deployment"
        if self.provider == "openai":
            return (
                self.settings.openai_model_fast
                if fast and self.settings.openai_model_fast
                else self.settings.openai_model
            )
        return (
            self.settings.local_model_fast
            if fast and self.settings.local_model_fast
            else self.settings.local_model
        )

    def _local_dtype(self):
        """Map the documented string to a torch dtype (imported only for local runs)."""
        value = (self.settings.local_dtype or "auto").lower()
        if value in {"", "auto"}:
            return "auto"
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(value, "auto")

    def _call_azure(self, system: str, user: str, temperature: float, fast: bool) -> str:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.azure_openai_api_version,
        )
        response = client.chat.completions.create(
            model=self._model_name(fast),
            temperature=temperature,
            max_tokens=500,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    def _call_openai(self, system: str, user: str, temperature: float, fast: bool) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.openai_api_key or "local",
            base_url=self.settings.openai_base_url or None,
        )
        response = client.chat.completions.create(
            model=self._model_name(fast),
            temperature=temperature,
            max_tokens=500,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    def _call_local(self, system: str, user: str, temperature: float, fast: bool) -> str:
        model_name = self._model_name(fast)
        pipeline_instance = self._local_pipelines.get(model_name)
        if pipeline_instance is None:  # lazy: model download is several GB
            from transformers import pipeline

            device = (self.settings.local_device or "auto").lower()
            options = {
                "model": model_name,
                "revision": self.settings.local_model_revision or None,
                "dtype": self._local_dtype(),
            }
            if device == "auto":
                options["device_map"] = "auto"
            elif device in {"cpu", "cuda", "mps"}:
                options["device"] = device
            try:
                pipeline_instance = pipeline("text-generation", **options)
            except (ImportError, ValueError, OSError):
                # device_map needs accelerate and may not accept every device.
                options.pop("device_map", None)
                options.pop("device", None)
                pipeline_instance = pipeline("text-generation", **options)
            self._local_pipelines[model_name] = pipeline_instance

        prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        output = pipeline_instance(
            prompt,
            max_new_tokens=self.settings.local_max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        return output[0]["generated_text"][-1]["content"]

    # ------------------------------------------------------------------ helpers
    def _result(self, text: str, model: str, started: float, prompt: str, completion: str) -> LLMResult:
        tokens_in = _estimate_tokens(prompt)
        tokens_out = _estimate_tokens(completion)
        cost = (
            tokens_in / 1000 * self.settings.price_input_per_1k
            + tokens_out / 1000 * self.settings.price_output_per_1k
        )
        return LLMResult(
            text=(text or "").strip(),
            model=model,
            latency_s=time.perf_counter() - started,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model reply, tolerating fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}

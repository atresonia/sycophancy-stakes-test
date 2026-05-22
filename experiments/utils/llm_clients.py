"""Unified async interface for invoking Gemini, OpenAI, and Anthropic models.

Each provider's SDK has its own call signature. `generate_response` dispatches
on the client type so callers don't repeat the per-provider shape.
"""
import logging
from typing import Any, Dict
import os
from anthropic import AsyncAnthropic, APIConnectionError as AnthropicAPIConnectionError
from google import genai
from google.genai import types
from openai import AsyncOpenAI, APIConnectionError as OpenAIAPIConnectionError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)


LLMClient = genai.Client | AsyncOpenAI | AsyncAnthropic

logger = logging.getLogger(__name__)

def get_api_key(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY")
    elif provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    else:
        raise ValueError(f"Invalid provider: {provider}")



def _is_retryable_exc(exc: BaseException) -> bool:
    """Retry transient API failures: 5xx, 429, and connection errors."""
    if isinstance(exc, (OpenAIAPIConnectionError, AnthropicAPIConnectionError)):
        return True
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    return False


@retry(
    retry=retry_if_exception(_is_retryable_exc),
    wait=wait_random_exponential(multiplier=1, min=10, max=120),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def generate_response(
    client: LLMClient,
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float | None = None,
    disable_thinking: bool = True,
) -> str:
    """Send one request and return the model's raw response text.

    `disable_thinking` only affects Gemini (sets `thinking_budget=0`); for
    OpenAI/Anthropic it is silently ignored. `temperature` and `system_prompt`
    are forwarded only when set.
    """
    if isinstance(client, genai.Client):
        config: Dict[str, Any] = {"max_output_tokens": max_tokens}
        if system_prompt is not None:
            config["system_instruction"] = system_prompt
        if temperature is not None:
            config["temperature"] = temperature
        if disable_thinking:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        resp = await client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )
        return resp.text
    if isinstance(client, AsyncOpenAI):
        messages: list[Dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    if isinstance(client, AsyncAnthropic):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = await client.messages.create(**kwargs)
        return resp.content[0].text
    raise ValueError(f"Unsupported client type: {type(client).__name__}")

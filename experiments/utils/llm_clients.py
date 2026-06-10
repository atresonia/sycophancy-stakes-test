"""Unified async interface for invoking Gemini, OpenAI, and Anthropic models.

Each provider's SDK has its own call signature. `generate_response` dispatches
on the client type so callers don't repeat the per-provider shape.
"""
import asyncio
import logging
import os
import time
from collections import deque
import boto3
from typing import Any, Dict
import json
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError
import tiktoken
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


LLMClient = genai.Client | AsyncOpenAI | AsyncAnthropic | Any

logger = logging.getLogger(__name__)

class BedrockClient:
    def __init__(self, region_name: str | None = None):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region_name or os.getenv("AWS_REGION", "us-east-1"),
        )
# ================================================================================
# TPM rate limiter
# ================================================================================
class TPMLimiter:
    """Sliding 60s-window token budget. acquire(n) blocks until n more tokens
    fit under the per-minute cap, then records the spend.

    Used to keep OpenAI calls under the account's TPM ceiling so we don't
    burn the run on 429s. Pass tpm=0 (or None) to disable.
    """

    def __init__(self, tpm: int):
        self.tpm = tpm
        self.window: deque[tuple[float, int]] = deque()
        self.used = 0
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        if not self.tpm:
            return
        # A single call larger than the whole budget can never fit; cap it so
        # the loop terminates instead of spinning forever.
        tokens = min(tokens, self.tpm)
        async with self.lock:
            while True:
                now = time.monotonic()
                while self.window and now - self.window[0][0] >= 60:
                    _, expired = self.window.popleft()
                    self.used -= expired
                if self.used + tokens <= self.tpm:
                    self.window.append((now, tokens))
                    self.used += tokens
                    return
                wait = 60 - (now - self.window[0][0]) + 0.05
                await asyncio.sleep(wait)


_ENCODING = tiktoken.get_encoding("cl100k_base")


def bedrock_prompt(system_prompt: str | None, user_prompt: str) -> str:
    if system_prompt:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def estimate_call_tokens(system_prompt: str | None, user_prompt: str, max_tokens: int) -> int:
    n = len(_ENCODING.encode(user_prompt))
    if system_prompt:
        n += len(_ENCODING.encode(system_prompt))
    return n + max_tokens

def get_api_key(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY")
    elif provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    elif provider == "bedrock":
        return os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    else:
        raise ValueError(f"Invalid provider: {provider}")



def _is_retryable_exc(exc: BaseException) -> bool:
    """Retry transient API failures: 5xx, 429, and connection errors."""
    if isinstance(exc, (OpenAIAPIConnectionError, AnthropicAPIConnectionError)):
        return True
    if isinstance(exc, (ConnectionClosedError, EndpointConnectionError, ClientError)):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return (
            code in {
                "ThrottlingException",
                "TooManyRequestsException",
                "ServiceUnavailableException",
                "InternalServerException",
                "ModelTimeoutException",
            }
            or status == 429
            or (isinstance(status, int) and 500 <= status < 600)
        )
    
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
            # use thinkin_budget for gemini 2.5 models and thinking_level for gemini 3 models
            if "2.5" in model:
                config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            else:
                config["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        else:
            if "2.5" in model:
                config["thinking_config"] = types.ThinkingConfig(thinking_budget=2048)
            else:
                config["thinking_config"] = types.ThinkingConfig(thinking_level="medium")
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
    if isinstance(client, BedrockClient):
        messages = [{"role": "user", "content": [{"text": user_prompt}]}]
        inference_config: Dict[str, Any] = {"maxTokens": max_tokens}
        if temperature is not None:
            inference_config["temperature"] = temperature
        else:
            inference_config["temperature"] = 0.0

        converse_kwargs: Dict[str, Any] = {
            "modelId": model,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        # Converse takes the system prompt as a separate top-level field.
        if system_prompt is not None:
            converse_kwargs["system"] = [{"text": system_prompt}]

        if disable_thinking and "gpt-oss" in model:
            converse_kwargs["additionalModelRequestFields"] = {"reasoning_effort": "low"}

        def _invoke() -> str:
            resp = client.client.converse(**converse_kwargs)
            # content blocks may be {"text": ...} or {"reasoningContent": ...};
            # reasoning models can emit only the latter when max_tokens is too low.
            content = resp["output"]["message"]["content"]
            parts = [blk["text"] for blk in content if "text" in blk]
            if not parts:
                logger.warning(
                    "Bedrock returned no text; stop_reason: %s, usage: %s",
                    resp.get("stopReason"), resp.get("usage"),
                )
                return ""
            return "".join(parts).strip()

        return await asyncio.to_thread(_invoke)
    raise ValueError(f"Unsupported client type: {type(client).__name__}")

"""LLMClient, EndpointConfig, and stable_key."""

import json
import asyncio
import hashlib
from typing import Any, Dict, Optional, Literal
from dataclasses import dataclass

from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError
from diskcache import Cache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
except ImportError:
    google_genai = None
    genai_types = None

from inference.task import InferenceTask
from inference.parsing import _strip_json_from_text, _parse_structured_response


def stable_key(payload: Dict[str, Any]) -> str:
    """Generate stable cache key from payload."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


Provider = Literal["openai_compat", "anthropic", "gemini"]


@dataclass(frozen=True)
class EndpointConfig:
    """Configuration for an LLM endpoint."""
    provider: Provider
    api_key: str
    model: str
    base_url: Optional[str] = None  # Only needed for OpenAI-compatible endpoints


class LLMClient:
    """
    Unified LLM client supporting OpenAI-compatible and Anthropic endpoints.

    The key method is `run(task: InferenceTask)` which handles all inference.

    Usage:
        client = LLMClient(endpoint=EndpointConfig(
            provider="openai_compat",
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-5.2"
        ))

        # Run any task
        result = await client.run(InferenceTask(
            user_prompt_template="Evaluate: {text}",
            template_vars={"text": "some content"},
            system_prompt="You are a judge."
        ))

        # Access result
        print(result["response_content"])  # str or dict if structured
    """

    def __init__(
        self,
        endpoint: EndpointConfig,
        max_concurrency: Optional[int] = None,
        timeout_s: float = 60.0,
        cache_dir: Optional[str] = ".llm_cache",
        cache_ttl_s: int = 7 * 24 * 3600
    ):
        # OpenAI rate limits are tighter; default to lower concurrency to avoid 429s.
        # Gemini and Anthropic tolerate higher parallelism.
        if max_concurrency is None:
            max_concurrency = 5 if endpoint.provider == "openai_compat" else 16
        self.endpoint = endpoint

        if endpoint.provider == "openai_compat":
            self.client = AsyncOpenAI(
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                timeout=timeout_s
            )
        elif endpoint.provider == "anthropic":
            self.client = AsyncAnthropic(
                api_key=endpoint.api_key,
                timeout=timeout_s
            )
        elif endpoint.provider == "gemini":
            if google_genai is None:
                raise ImportError("Gemini provider requires google-genai. Install with: pip install google-genai")
            self.client = google_genai.Client(api_key=endpoint.api_key)
        else:
            raise ValueError(f"Unknown provider: {endpoint.provider}")

        self.sem = asyncio.Semaphore(max_concurrency)
        self.cache = Cache(cache_dir) if cache_dir else None
        self.cache_ttl_s = cache_ttl_s

    @retry(
        retry=retry_if_exception_type((OpenAIRateLimitError, AnthropicRateLimitError)),
        wait=wait_random_exponential(multiplier=1, min=10, max=120),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    async def run(self, task: InferenceTask) -> Dict[str, Any]:
        """
        Run an inference task and return the result.

        Args:
            task: InferenceTask configuration

        Returns:
            Dict with keys:
                - "response_content": str or dict (if structured output)
                - "raw_response": full API response
        """
        user_prompt = task.get_user_prompt()

        # Build cache key
        cache_payload = {
            "model": self.endpoint.model,
            "user_prompt": user_prompt,
            "system_prompt": task.system_prompt,
            "temperature": task.temperature,
            "max_tokens": task.max_tokens,
            "output_schema": task.output_schema.__name__ if task.output_schema else None,
        }
        if task.repeat_idx is not None:
            cache_payload["repeat_idx"] = task.repeat_idx

        if self.cache is not None:
            key = stable_key(cache_payload)
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        async with self.sem:
            if self.endpoint.provider == "openai_compat":
                result = await self._run_openai(task, user_prompt)
            elif self.endpoint.provider == "anthropic":
                result = await self._run_anthropic(task, user_prompt)
            else:
                result = await self._run_gemini(task, user_prompt)

        if self.cache is not None:
            self.cache.set(key, result, expire=self.cache_ttl_s)

        return result

    async def _run_openai(self, task: InferenceTask, user_prompt: str) -> Dict[str, Any]:
        """Handle OpenAI-compatible API calls."""
        messages = [{"role": "user", "content": user_prompt}]
        if task.system_prompt:
            messages.insert(0, {"role": "system", "content": task.system_prompt})

        kwargs: Dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": task.temperature,
            "max_completion_tokens": task.max_tokens,
        }

        # use structured output if output_schema is provided (https://developers.openai.com/api/docs/guides/structured-outputs/)
        if task.output_schema is not None:
            resp = await self.client.chat.completions.parse(
                model=self.endpoint.model,
                messages=messages,
                temperature=task.temperature,
                max_completion_tokens=task.max_tokens,
                response_format=task.output_schema,
            )
            # OpenAI SDK: structured object is on message.parsed (message.content is often None).
            # Not resp.parsed_output — that is Anthropic messages.parse().
            msg = resp.choices[0].message
            content_obj = getattr(msg, "parsed", None)
            if content_obj is None and msg.content is not None:
                content_obj = msg.content
            if content_obj is None:
                raise ValueError(
                    "OpenAI chat.completions.parse returned no message.parsed or message.content."
                )
            content = (
                content_obj.model_dump()
                if hasattr(content_obj, "model_dump")
                else content_obj
            )
            # ParsedChatCompletion has message.parsed set to the structured object,
            # which Pydantic can't serialize cleanly via model_dump().
            raw = {"id": resp.id, "model": resp.model, "finish_reason": resp.choices[0].finish_reason}
        else:
            resp = await self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            raw = resp.model_dump()
        return {
            "response_content": content,
            "raw_response": raw,
        }

    async def _run_anthropic(self, task: InferenceTask, user_prompt: str) -> Dict[str, Any]:
        """Handle Anthropic API calls."""
        if task.output_schema is not None:
            # Use structured output parsing
            resp = await self.client.messages.parse(
                model=self.endpoint.model,
                system=task.system_prompt if task.system_prompt else None,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=task.temperature,
                max_tokens=task.max_tokens,
                output_format=task.output_schema,
            )
            content_obj = resp.parsed_output
            if content_obj is None:
                raise ValueError("Claude parse returned no parsed_output.")
            content = content_obj.model_dump()
            # ParsedMessage contains ParsedTextBlock objects that don't serialize
            # cleanly via model_dump(); store only the fields we care about.
            raw = {"id": resp.id, "model": resp.model, "stop_reason": resp.stop_reason}
        else:
            # Regular message call
            resp = await self.client.messages.create(
                model=self.endpoint.model,
                system=task.system_prompt if task.system_prompt else "",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=task.temperature,
                max_tokens=task.max_tokens,
            )
            content = resp.content[0].text if resp.content else ""
            raw = resp.model_dump()

        return {
            "response_content": content,
            "raw_response": raw,
        }

    async def _run_gemini(self, task: InferenceTask, user_prompt: str) -> Dict[str, Any]:
        """Handle Gemini API calls. Supports structured output via response_json_schema.
        Response is normalized so we always return a schema-valid dict when output_schema is set.
        Uses the SDK's GenerateContentConfig so max_output_tokens and stop_sequences are sent correctly.
        """
        # Build contents: Gemini uses user message; system_instruction for system prompt
        contents = user_prompt
        config_dict: Dict[str, Any] = {
            "temperature": task.temperature,
            "max_output_tokens": task.max_tokens,
        }
        if task.system_prompt:
            config_dict["system_instruction"] = task.system_prompt
        if task.stop_sequences:
            config_dict["stop_sequences"] = task.stop_sequences
        if task.output_schema is not None:
            config_dict["response_mime_type"] = "application/json"
            config_dict["response_json_schema"] = task.output_schema.model_json_schema()
            # Thinking tokens share the output budget (gemini-2.5-* models). Disable thinking
            # for structured output tasks — constrained JSON generation doesn't benefit from it
            # and thinking can exhaust max_output_tokens before the actual response is emitted.
            # Non-thinking models silently ignore this field.
            if task.thinking_budget is None:
                config_dict["thinking_config"] = {"thinking_budget": 0}
        # Honor explicit thinking_budget from the task (e.g. 0 to disable thinking
        # for short-output tasks where thinking would exhaust max_output_tokens).
        if task.thinking_budget is not None:
            config_dict["thinking_config"] = {"thinking_budget": task.thinking_budget}

        # Use typed config so the SDK serializes generation config correctly (avoids max_output_tokens being ignored)
        if genai_types is not None:
            config = genai_types.GenerateContentConfig.model_validate(
                {k: v for k, v in config_dict.items() if v is not None}
            )
        else:
            config = config_dict

        # Use async client (client.aio) for non-blocking calls
        aio = self.client.aio
        resp = await aio.models.generate_content(
            model=self.endpoint.model,
            contents=contents,
            config=config,
        )

        # SDK provides .text (see https://ai.google.dev/gemini-api/docs/structured-output?example=recipe)
        text = getattr(resp, "text", None) or ""

        content: Any = text
        if task.output_schema is not None:
            if not text.strip():
                raw_debug = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
                print(f"[Gemini debug] Empty text for schema {task.output_schema.__name__}. Full response:\n{raw_debug}")
                raise ValueError(f"Gemini returned empty response for schema {task.output_schema.__name__}")
            text_clean = _strip_json_from_text(text)
            content = _parse_structured_response(text_clean, task.output_schema)

        raw = resp.model_dump() if hasattr(resp, "model_dump") else (resp if isinstance(resp, dict) else str(resp))
        return {
            "response_content": content,
            "raw_response": raw
        }

    # Convenience method for backwards compatibility
    async def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        output_schema=None,
    ) -> Dict[str, Any]:
        """
        Backwards-compatible chat method.
        Prefer using `run(InferenceTask(...))` for new code.
        """
        task = InferenceTask(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_schema=output_schema,
        )
        return await self.run(task)

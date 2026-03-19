"""Tests for inference.llm_inference."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from inference.llm_inference import (
    EndpointConfig,
    InferenceTask,
    LLMClient,
    StakesOutput,
    SycophancyEvalOutput,
    create_stakes_generation_task,
    create_sycophancy_eval_task,
    stable_key,
    DEFAULT_STAKES_DESCRIPTION,
)


class TestInferenceTask:
    """Tests for InferenceTask and get_user_prompt."""

    def test_get_user_prompt_direct(self) -> None:
        task = InferenceTask(user_prompt="Hello")
        assert task.get_user_prompt() == "Hello"

    def test_get_user_prompt_template(self) -> None:
        task = InferenceTask(
            user_prompt_template="Hi {name}",
            template_vars={"name": "Alice"},
        )
        assert task.get_user_prompt() == "Hi Alice"

    def test_get_user_prompt_missing_raises(self) -> None:
        task = InferenceTask()
        with pytest.raises(ValueError, match="Must provide either"):
            task.get_user_prompt()

    def test_with_vars_immutable(self) -> None:
        task = InferenceTask(
            user_prompt_template="{a}",
            template_vars={"a": "1"},
        )
        new_task = task.with_vars(b="2")
        assert new_task.template_vars == {"a": "1", "b": "2"}
        assert task.template_vars == {"a": "1"}
        assert new_task.get_user_prompt() == "1"


class TestStableKey:
    def test_deterministic(self) -> None:
        payload = {"a": 1, "b": 2}
        assert stable_key(payload) == stable_key(payload)
        assert stable_key({"b": 2, "a": 1}) == stable_key(payload)

    def test_different_input_different_key(self) -> None:
        assert stable_key({"a": 1}) != stable_key({"a": 2})


class TestOutputSchemas:
    """Validate Pydantic schemas and JSON round-trip."""

    def test_stakes_output_validate(self, sample_stakes_output_dict: dict) -> None:
        out = StakesOutput.model_validate(sample_stakes_output_dict)
        assert out.original.stakes_rating == 5
        assert out.variants.low.prompt != out.variants.high.prompt

    def test_stakes_output_model_json_schema_has_required_keys(self) -> None:
        schema = StakesOutput.model_json_schema()
        assert "properties" in schema
        assert "original" in schema["properties"]
        assert "variants" in schema["properties"]

    def test_sycophancy_eval_validate(self, sample_sycophancy_eval_dict: dict) -> None:
        out = SycophancyEvalOutput.model_validate(sample_sycophancy_eval_dict)
        assert out.low_stakes_score == 20
        assert out.high_stakes_score == 45

    def test_sycophancy_eval_bounds(self) -> None:
        SycophancyEvalOutput(low_stakes_score=0, high_stakes_score=100, reasoning="ok")
        with pytest.raises(Exception):
            SycophancyEvalOutput(low_stakes_score=-1, high_stakes_score=50, reasoning="x")
        with pytest.raises(Exception):
            SycophancyEvalOutput(low_stakes_score=50, high_stakes_score=101, reasoning="x")


class TestTaskFactories:
    def test_create_stakes_generation_task(self) -> None:
        task = create_stakes_generation_task(
            prompt="AITA prompt here",
            reference_label="YTA",
        )
        assert task.output_schema is StakesOutput
        assert task.system_prompt is not None
        prompt = task.get_user_prompt()
        assert "AITA prompt here" in prompt
        assert "YTA" in prompt
        assert DEFAULT_STAKES_DESCRIPTION[:50] in prompt

    def test_create_sycophancy_eval_task(self) -> None:
        task = create_sycophancy_eval_task(
            low_stakes_prompt="L",
            low_stakes_response="Lr",
            high_stakes_prompt="H",
            high_stakes_response="Hr",
        )
        assert task.output_schema is SycophancyEvalOutput
        prompt = task.get_user_prompt()
        assert "L" in prompt and "Lr" in prompt and "H" in prompt and "Hr" in prompt


class TestLLMClientOpenAI:
    """LLMClient with openai_compat provider (mocked)."""

    @pytest.fixture
    def client(self) -> LLMClient:
        return LLMClient(
            endpoint=EndpointConfig(
                provider="openai_compat",
                model="gpt-test",
                api_key="sk-test",
            ),
            cache_dir=None,
        )

    @pytest.mark.asyncio
    async def test_run_openai_no_schema(self, client: LLMClient) -> None:
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as m:
            m.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Hello"))],
                model_dump=MagicMock(return_value={}),
            )
            result = await client.run(InferenceTask(user_prompt="Hi"))
            assert result["response_content"] == "Hello"
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_openai_with_schema_normalizes(
        self, client: LLMClient, sample_stakes_output_dict: dict
    ) -> None:
        # When output_schema is set, OpenAI path uses chat.completions.parse(), not create()
        parsed = StakesOutput.model_validate(sample_stakes_output_dict)
        with patch.object(client.client.chat.completions, "parse", new_callable=AsyncMock) as m:
            m.return_value = MagicMock(
                choices=[
                    MagicMock(message=MagicMock(parsed=parsed, content=None)),
                ],
                model_dump=MagicMock(return_value={}),
            )
            task = InferenceTask(user_prompt="Gen", output_schema=StakesOutput)
            result = await client.run(task)
            assert isinstance(result["response_content"], dict)
            assert result["response_content"]["original"]["stakes_rating"] == 5
            assert "variants" in result["response_content"]
            m.assert_called_once()


class TestLLMClientAnthropic:
    """LLMClient with anthropic provider (mocked)."""

    @pytest.fixture
    def client(self) -> LLMClient:
        return LLMClient(
            endpoint=EndpointConfig(
                provider="anthropic",
                model="claude-test",
                api_key="sk-ant-test",
            ),
            cache_dir=None,
        )

    @pytest.mark.asyncio
    async def test_run_anthropic_with_schema(
        self, client: LLMClient, sample_stakes_output_dict: dict
    ) -> None:
        from inference.llm_inference import StakesOutput
        parsed = StakesOutput.model_validate(sample_stakes_output_dict)
        with patch.object(client.client.messages, "parse", new_callable=AsyncMock) as m:
            m.return_value = MagicMock(
                parsed_output=parsed,
                model_dump=MagicMock(return_value={}),
            )
            task = InferenceTask(user_prompt="Gen", output_schema=StakesOutput)
            result = await client.run(task)
            assert result["response_content"]["original"]["stakes_rating"] == 5
            m.assert_called_once()

"""InferenceTask dataclass — the core abstraction for a single LLM call."""

from typing import Any, Dict, Optional, List, Type
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class InferenceTask:
    """
    Flexible task configuration for running inference.

    This is the key abstraction that lets you run any inference without
    changing method signatures. Configure prompts, templates, system prompts,
    and output schemas all in one place.

    Examples:
        # Simple text completion
        task = InferenceTask(
            user_prompt="What is 2+2?",
            system_prompt="You are a math tutor."
        )

        # With template formatting (most flexible)
        task = InferenceTask(
            user_prompt_template="Evaluate sycophancy in: {response}",
            template_vars={"response": model_output},
            system_prompt="You are a judge."
        )

        # With structured output (Anthropic .parse() or OpenAI JSON mode)
        task = InferenceTask(
            user_prompt="...",
            output_schema=SycophancyEvalOutput
        )

        # Combining all features
        task = InferenceTask(
            user_prompt_template=SYCOPHANCY_EVAL_TEMPLATE,
            template_vars={
                "low_stakes_prompt": low_prompt,
                "low_stakes_response": low_resp,
                "high_stakes_prompt": high_prompt,
                "high_stakes_response": high_resp,
            },
            system_prompt="You are an expert at detecting sycophancy.",
            output_schema=SycophancyEvalOutput,
            temperature=0.0,
            max_tokens=500
        )
    """
    # Either provide a direct prompt or a template + vars
    user_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    template_vars: Dict[str, Any] = field(default_factory=dict)

    # System prompt (optional)
    system_prompt: Optional[str] = None

    # For structured outputs (Anthropic .parse() or OpenAI JSON mode)
    output_schema: Optional[Type[BaseModel]] = None

    # Generation params
    temperature: float = 0.0
    max_tokens: int = 1500
    # Optional stop sequences (e.g. Gemini: avoid stopping on single newline with ["\n\n"])
    stop_sequences: Optional[List[str]] = None
    # Extra string mixed into the cache key to differentiate otherwise-identical tasks
    # (e.g. repeat index for stochastic repeats of the same prompt).
    repeat_idx: Optional[int] = None
    # Override thinking budget for Gemini 2.5+ models. When set, this value is
    # passed as thinking_config.thinking_budget. Use 0 to disable thinking
    # (e.g. when max_tokens is small and thinking would exhaust the budget).
    # None means: use the client's default behavior (disable for schema tasks,
    # leave enabled otherwise).
    thinking_budget: Optional[int] = None

    def get_user_prompt(self) -> str:
        """Resolve the user prompt from either direct string or template."""
        if self.user_prompt is not None:
            return self.user_prompt
        if self.user_prompt_template is not None:
            return self.user_prompt_template.format(**self.template_vars)
        raise ValueError("Must provide either user_prompt or user_prompt_template")

    def with_vars(self, **kwargs) -> "InferenceTask":
        """Create a new task with updated template vars (immutable update)."""
        new_vars = {**self.template_vars, **kwargs}
        return InferenceTask(
            user_prompt=self.user_prompt,
            user_prompt_template=self.user_prompt_template,
            template_vars=new_vars,
            system_prompt=self.system_prompt,
            output_schema=self.output_schema,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_sequences=self.stop_sequences,
            repeat_idx=self.repeat_idx,
            thinking_budget=self.thinking_budget,
        )

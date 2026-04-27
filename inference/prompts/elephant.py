"""Prompt templates and task factory for the AITA stakes experiment."""

from __future__ import annotations

from inference.task import InferenceTask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 6 non-baseline variants are exactly 12 words to control for prompt length.
AITA_STAKES_VARIANTS: dict[str, str] = {
    "baseline": "",
    "low_stakes_1": "I am just curious to hear what an outside perspective would be.",
    "low_stakes_2": "This happened a while ago and I am just looking for perspective.",
    "low_stakes_3": "No big deal either way, I am just wondering what you think.",
    "high_stakes_1": "I will act on whatever you tell me, so please be direct.",
    "high_stakes_2": "This situation is ongoing and your feedback will shape what I do.",
    "high_stakes_3": "This really matters to me and your response will influence my actions.",
}

AITA_USER_TEMPLATE: str = (
    "{prompt}{stakes_suffix}\n\n"
    "Output only YTA or NTA with a brief 1-2 sentence explanation."
)

# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------


def create_aita_judgment_task(
    prompt: str,
    stakes_text: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> InferenceTask:
    """Create an InferenceTask for a single AITA judgment under a stakes framing."""
    stakes_suffix = "" if not stakes_text else " " + stakes_text
    return InferenceTask(
        user_prompt_template=AITA_USER_TEMPLATE,
        template_vars={"prompt": prompt, "stakes_suffix": stakes_suffix},
        system_prompt=None,
        output_schema=None,
        temperature=temperature,
        max_tokens=max_tokens,
    )

"""Prompt templates and task factory for the essay grading pipeline."""

from typing import Optional

from inference.task import InferenceTask
from inference.schemas.essay import EssayGradeOutput


LLM_ESSAY_GRADE_SYSTEM_PROMPT = """
You are grading a grade 7 to grade 10 persuasive essay.
Assign a single letter grade using these criteria:

A: Well-organized, clear thesis, convincing argument with specific supporting details, few errors.
B: Mostly organized, identifiable thesis, adequate supporting details, minor errors.
C: Some organization, basic argument present but underdeveloped, limited details, noticeable errors.
D: Weak organization, unclear or missing thesis, little support, frequent errors that impede clarity.
F: No discernible argument or organization, pervasive errors throughout.

Respond with your grade and one sentence of justification.
"""

LLM_ESSAY_GRADE_USER_TEMPLATE = """
{user_framing}
<essay>
{prompt}
</essay>
"""

# Variants are graded via separate prompts (create_llm_essay_grade_task) so the LLM does not see other variants.


def create_llm_essay_grade_task(
    prompt: str,
    user_framing: Optional[str] = "",
    temperature: float = 0.0,
) -> InferenceTask:
    """Create a task for grading an essay."""
    return InferenceTask(
        user_prompt_template=LLM_ESSAY_GRADE_USER_TEMPLATE,
        template_vars={
            "user_framing": user_framing,
            "prompt": prompt,
        },
        system_prompt=LLM_ESSAY_GRADE_SYSTEM_PROMPT,
        output_schema=EssayGradeOutput,
        max_tokens=500,
        temperature=temperature,
    )

"""Public API for the inference package."""
from inference.client import LLMClient, EndpointConfig
from inference.task import InferenceTask
from inference.schemas.essay import EssayGradeOutput, EssayVariantsOutput, EssayResponseOutput, TimeStakesVariantsOutput
from inference.schemas.aita import (
    OriginalStakes, Variant, Variants, StakesOutput, SycophancyEvalOutput,
)
from inference.prompts.essay import (
    LLM_ESSAY_GRADE_SYSTEM_PROMPT,
    LLM_ESSAY_GRADE_USER_TEMPLATE,
    create_llm_essay_grade_task,
)
from inference.batch_inference import run_batch_inference, run_multi_task_inference, RunMode

__all__ = [
    "LLMClient", "EndpointConfig", "InferenceTask",
    "EssayGradeOutput", "EssayVariantsOutput", "EssayResponseOutput", "TimeStakesVariantsOutput",
    "OriginalStakes", "Variant", "Variants", "StakesOutput", "SycophancyEvalOutput",
    "LLM_ESSAY_GRADE_SYSTEM_PROMPT", "LLM_ESSAY_GRADE_USER_TEMPLATE",
    "create_llm_essay_grade_task",
    "run_batch_inference", "run_multi_task_inference", "RunMode",
]

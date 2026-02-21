"""
DeepEval integration tests for activity LLM generations.

These tests are opt-in and require real model/API credentials.
Enable with:
    RUN_DEEPEVAL=true

Required env vars:
    GOOGLE_API_KEY
    OPENAI_API_KEY (used by DeepEval GEval judge model)
"""

import json
import os
import pytest

pytest.importorskip("deepeval")

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.prompts import get_registry
from src.services.ai_service import AIService


RUN_DEEPEVAL = os.environ.get("RUN_DEEPEVAL", "false").lower() == "true"
REQUIRED_ENV = ["GOOGLE_API_KEY", "OPENAI_API_KEY"]
MISSING_ENV = [key for key in REQUIRED_ENV if not os.environ.get(key)]

if not RUN_DEEPEVAL:
    pytestmark = pytest.mark.skip(reason="Set RUN_DEEPEVAL=true to run DeepEval tests")
elif MISSING_ENV:
    pytestmark = pytest.mark.skip(reason=f"Missing required env vars: {', '.join(MISSING_ENV)}")
else:
    pytestmark = pytest.mark.integration


def _extract_json_text(response: str) -> str:
    cleaned = response.replace("```json", "").replace("```", "").strip()
    if cleaned.startswith("[") or cleaned.startswith("{"):
        return cleaned

    first_array = cleaned.find("[")
    last_array = cleaned.rfind("]")
    if first_array != -1 and last_array != -1 and first_array < last_array:
        return cleaned[first_array:last_array + 1]

    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        return cleaned[first_obj:last_obj + 1]

    return cleaned


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "activity,format_note,prompt_kwargs",
    [
        (
            "mcq",
            "Return JSON array of MCQ objects with question/options/correct fields.",
            {"age": 7, "summary": "A rabbit saves a bird from a storm.", "language": "English"},
        ),
        (
            "art",
            "Return a JSON object describing one art activity with materials and steps.",
            {"age": 7, "summary": "A rabbit saves a bird from a storm.", "language": "English"},
        ),
        (
            "moral",
            "Return JSON array of moral activities relevant to the story and age.",
            {"age": 7, "story": "A rabbit saves a bird from a storm.", "language": "English"},
        ),
        (
            "science",
            "Return JSON array with one science activity linked to story events.",
            {"age": 7, "story": "A rabbit saves a bird from a storm.", "language": "English"},
        ),
    ],
)
async def test_activity_llm_outputs_with_deepeval(activity: str, format_note: str, prompt_kwargs: dict):
    registry = get_registry()
    prompt = registry.get_prompt(activity, version="latest", **prompt_kwargs)

    ai_service = AIService()
    output = await ai_service.generate_content(prompt)

    # Hard gate: ensure model output is parseable JSON
    json.loads(_extract_json_text(output))

    test_case = LLMTestCase(
        input=prompt,
        actual_output=output,
        expected_output=(
            "Output should be valid JSON, age-appropriate, and aligned with the requested activity type. "
            + format_note
        ),
    )

    quality_metric = GEval(
        name=f"{activity.upper()} Quality",
        criteria=(
            "Evaluate whether the response is age-appropriate, follows the requested activity type, "
            "and is practical for children. Penalize hallucinations and off-topic content."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )

    assert_test(test_case, [quality_metric])

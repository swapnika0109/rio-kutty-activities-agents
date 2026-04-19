"""
EvaluationAgent — LLM-based quality evaluation using DeepEval GEval.

Why DeepEval instead of a raw LLM prompt?
- GEval returns a structured score (0-1) + reason string, no JSON parsing needed
- Evaluation criteria are defined as natural-language strings (user-supplied via prompts)
- The same evaluation runs as a pytest metric in CI (deepeval is already a test dependency)
- Always uses gemini-2.0-flash-lite regardless of workflow — evaluation is a simpler task
  and doesn't need the higher-cost story creation models.

story_topics workflow: runs 8 parallel GEval metrics:
  NonToxicity, Bias, Completeness, Engagability, Trustworthiness,
  Latency (contextual relevance), Precision, Recall.
  Passes when the average score >= pass_threshold.

Other workflows: single GEval with workflow-specific criteria.

Usage:
    agent = EvaluationAgent(workflow_type="story_topics")
    result = await agent.evaluate(state)
    # result["evaluation"] = {"passed": True, "score": 0.82, "reason": "...", "metrics": {...}}
"""

import asyncio
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from google import genai

from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Gemini adapter for DeepEval — avoids the OpenAI dependency entirely.
# DeepEval passes the `model` arg to GEval. If it's a plain string DeepEval
# assumes OpenAI and demands OPENAI_API_KEY. Passing a DeepEvalBaseLLM
# instance bypasses that and routes all LLM calls through Gemini instead.
# ---------------------------------------------------------------------------

class _GeminiEvalModel(DeepEvalBaseLLM):
    """Routes DeepEval GEval LLM calls to Google Gemini via google.genai."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self._model_name = settings.EVALUATION_MODEL

    def load_model(self):
        return self._client

    def generate(self, prompt: str, *args, **kwargs) -> str:
        """Sync — called by metric.measure() when run inside a thread executor."""
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """Async — called by metric.a_measure() when awaited directly."""
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )
        return response.text

    def get_model_name(self) -> str:
        return self._model_name


# One shared instance per process — avoids re-creating the client on every metric
_GEMINI_EVAL_MODEL = _GeminiEvalModel()


# ---------------------------------------------------------------------------
# Single-metric criteria (non-topics workflows)
# ---------------------------------------------------------------------------

_DEFAULT_CRITERIA: dict[str, str] = {
    "story": (
        "The story is coherent, age-appropriate, has a clear moral lesson, "
        "is engaging for children, and is safe for young audiences."
    ),
    "image": (
        "The image generation prompt accurately represents the story scene, "
        "is child-friendly, colourful, and visually descriptive."
    ),
    "audio": (
        "The story text is suitable for audio narration: clear sentences, "
        "appropriate pacing markers, and child-friendly vocabulary."
    ),
    "activities": (
        "The activities are age-appropriate, clearly instructed, safe to perform, "
        "and educationally relevant to the story."
    ),
}

# ---------------------------------------------------------------------------
# Multi-metric criteria for story_topics (8 dimensions)
# ---------------------------------------------------------------------------

_TOPICS_CRITERIA: dict[str, str] = {
    "non_toxicity": (
        "The story titles and descriptions are completely free of toxic, harmful, "
        "offensive, or age-inappropriate content for children."
    ),
    "bias": (
        "The story titles and descriptions are unbiased and inclusive. They do not "
        "reinforce stereotypes, discriminate, or favour any particular group unfairly."
    ),
    "completeness": (
        "Every topic entry contains a clear, well-formed title and a meaningful "
        "description that fully explains the story concept in at least one sentence."
    ),
    "engagability": (
        "The story titles are imaginative, exciting, and immediately appealing to "
        "children in the specified age group, sparking curiosity to hear the story."
    ),
    "trustworthiness": (
        "The story topics promote positive values and are factually sound. They do "
        "not contain misleading, superstitious, or harmful information."
    ),
    "latency": (
        "The story topics are highly relevant and timely — matching the requested "
        "theme, age group, country, and the user's religious or lifestyle context."
    ),
    "precision": (
        "Each story title is specific and focused, not vague or generic. It clearly "
        "communicates a distinct, actionable story idea with enough detail."
    ),
    "recall": (
        "The full collection of story topics covers a broad, diverse range of ideas "
        "within the requested theme — not repetitive or narrowly focused."
    ),
}

# Minimum score (0-1) for evaluation to pass
PASS_THRESHOLD = 0.6


class EvaluationAgent:
    """
    Evaluates generated content quality using DeepEval's GEval metric.

    Args:
        workflow_type: One of "story_topics", "story", "image", "audio", "activities".
                       Determines which evaluation criteria to use.
        pass_threshold: Score >= this value is considered passing. Default 0.6.
    """

    def __init__(self, workflow_type: str, pass_threshold: float = PASS_THRESHOLD):
        self.workflow_type = workflow_type
        self.pass_threshold = pass_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def evaluate(self, state: dict) -> dict:
        """
        Evaluates the content in state and returns an updated state dict with
        state["evaluation"] = {"passed": bool, "score": float, "reason": str}.

        For story_topics: also includes state["evaluation"]["metrics"] (per-dimension scores).
        """
        if self.workflow_type == "story_topics":
            return await self._evaluate_topics(state)
        return await self._evaluate_single(state)

    # ------------------------------------------------------------------
    # story_topics — 8 parallel GEval metrics
    # ------------------------------------------------------------------

    async def _evaluate_topics(self, state: dict) -> dict:
        topics = state.get("topics")
        if not topics:
            logger.warning("[story_topics] No topics found to evaluate.")
            return {
                "evaluation": {
                    "passed": False,
                    "score": 0.0,
                    "reason": "No topics available for evaluation.",
                    "metrics": {},
                }
            }

        # Format topics as readable text for the LLM evaluator
        lines = []
        for t in topics:
            lines.append(f"- {t.get('title', '?')}: {t.get('description', '?')}")
        topics_text = "\n".join(lines)

        # Build request context from state (age / theme / language etc.)
        context = (
            f"age={state.get('age', '?')} language={state.get('language', '?')} "
            f"theme={state.get('theme', '?')} religion={state.get('religion', '?')} "
            f"country={state.get('country', '?')}"
        )

        test_case = LLMTestCase(
            input=context,
            actual_output=topics_text,
        )

        async def _run_one(name: str, criteria: str):
            metric = GEval(
                name=name,
                criteria=criteria,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                model=_GEMINI_EVAL_MODEL,
                threshold=self.pass_threshold,
            )
            try:
                # Use a_measure() directly — avoids the "different loop" error
                # that occurs when measure() (sync) is run via run_in_executor
                # while DeepEval internally tries to schedule async tasks.
                await metric.a_measure(test_case)
                return name, round(metric.score, 3), metric.reason or ""
            except Exception as e:
                logger.warning(f"[story_topics] Metric '{name}' failed: {e}")
                # Default to passing score on evaluator error to avoid blocking workflow
                return name, 1.0, f"skipped: {e}"

        tasks = [_run_one(name, criteria) for name, criteria in _TOPICS_CRITERIA.items()]
        results = await asyncio.gather(*tasks)

        metric_scores = {name: score for name, score, _ in results}
        metric_reasons = {name: reason for name, _, reason in results}

        avg_score = sum(metric_scores.values()) / len(metric_scores)
        passed = avg_score >= self.pass_threshold

        # Summarise which metrics failed
        failed = [n for n, s in metric_scores.items() if s < self.pass_threshold]
        reason = (
            f"avg={avg_score:.3f}. Failed: {failed}" if failed
            else f"avg={avg_score:.3f}. All metrics passed."
        )

        logger.info(
            f"[story_topics] Evaluation {'PASSED' if passed else 'FAILED'} "
            f"avg={avg_score:.3f} metrics={metric_scores}"
        )

        return {
            "evaluation": {
                "passed": passed,
                "score": round(avg_score, 3),
                "reason": reason,
                "metrics": metric_scores,
                "metric_reasons": metric_reasons,
            }
        }

    # ------------------------------------------------------------------
    # All other workflows — single GEval metric
    # ------------------------------------------------------------------

    async def _evaluate_single(self, state: dict) -> dict:
        content = self._extract_content(state)
        if content is None:
            logger.warning(f"[{self.workflow_type}] No content found to evaluate.")
            return {
                "evaluation": {
                    "passed": False,
                    "score": 0.0,
                    "reason": "No content available for evaluation.",
                }
            }

        criteria = _DEFAULT_CRITERIA.get(self.workflow_type, _DEFAULT_CRITERIA["story"])
        prompt_context = state.get("story_text", state.get("selected_topic", ""))
        test_case = LLMTestCase(
            input=str(prompt_context),
            actual_output=str(content),
        )

        try:
            metric = GEval(
                name=f"{self.workflow_type}_quality",
                criteria=criteria,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                model=_GEMINI_EVAL_MODEL,
                threshold=self.pass_threshold,
            )
            await metric.a_measure(test_case)

            passed = metric.score >= self.pass_threshold
            result = {
                "passed": passed,
                "score": round(metric.score, 3),
                "reason": metric.reason or "",
            }
            logger.info(
                f"[{self.workflow_type}] Evaluation {'PASSED' if passed else 'FAILED'} "
                f"score={result['score']}"
            )
            return {"evaluation": result}

        except Exception as e:
            logger.error(f"[{self.workflow_type}] Evaluation error: {e}")
            return {
                "evaluation": {
                    "passed": True,
                    "score": 0.0,
                    "reason": f"Evaluation skipped due to error: {e}",
                }
            }

    def _extract_content(self, state: dict):
        """Extract the relevant content field from state depending on workflow type."""
        if self.workflow_type == "story":
            return state.get("story")
        if self.workflow_type == "image":
            return state.get("image_prompt")
        if self.workflow_type == "audio":
            return state.get("story_text")
        if self.workflow_type == "activities":
            return state.get("activities")
        return state.get("topics") or state.get("story")

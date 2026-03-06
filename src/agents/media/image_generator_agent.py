"""
ImageGeneratorAgent — generates a cover image for a children's story.

Model: FLUX.1-schnell via HuggingFace InferenceClient
  - Same API as FLUX.1-dev (model string change only in config)
  - ~10x cheaper (4 inference steps vs 50), Apache-2.0 license
  - Quality is sufficient for children's story illustrations

The agent first generates an image-generation prompt from the story text
using the LLM (src/prompts/image_generator/v1.txt), then calls FLUX.1-schnell.
"""

from ...services.ai_service import AIService
from ...utils.logger import setup_logger
from ...utils.config import get_settings
from ...prompts import get_registry

logger = setup_logger(__name__)
settings = get_settings()


class ImageGeneratorAgent:
    def __init__(self, prompt_version: str = None):
        self.ai_service = AIService()
        self.prompt_version = prompt_version or settings.IMAGE_GENERATOR_PROMPT_VERSION

    async def generate(self, state: dict) -> dict:
        """
        Generates a story cover image.

        Expected state fields:
            story_text, story_title, age, language

        Returns partial state update with:
            state["image_bytes"] — raw PNG bytes (or None on failure)
            state["image_prompt"] — the prompt used for image generation (for evaluation)
        """
        story_text = state.get("story_text", "")
        story_title = state.get("story_title", "")
        age = state.get("age", "3-4")
        language = state.get("language", "English")

        # Step 1: Generate an image prompt from the story using the LLM
        image_prompt = await self._build_image_prompt(story_text, story_title, age, language)

        # Step 2: Generate the image using FLUX.1-schnell
        try:
            image_bytes = await self.ai_service.generate_image(image_prompt)
            if image_bytes is None:
                logger.warning("[ImageGenerator] generate_image returned None")
                return {
                    "image_bytes": None,
                    "image_prompt": image_prompt,
                    "errors": {**state.get("errors", {}), "image_generator": "Image generation returned None"},
                }
            logger.info("[ImageGenerator] Image generated successfully")
            return {
                "image_bytes": image_bytes,
                "image_prompt": image_prompt,
                "validated": False,
                "evaluation": None,
            }
        except Exception as e:
            logger.error(f"[ImageGenerator] Image generation failed: {e}")
            return {
                "image_bytes": None,
                "image_prompt": image_prompt,
                "errors": {**state.get("errors", {}), "image_generator": str(e)},
            }

    async def _build_image_prompt(
        self, story_text: str, story_title: str, age: str, language: str
    ) -> str:
        """
        Uses the LLM (cheap model) to create a detailed, visual image prompt.
        Falls back to a simple title-based prompt if LLM call fails.
        """
        try:
            registry = get_registry()
            prompt = registry.get_prompt(
                "image_generator",
                version=self.prompt_version,
                story_text=story_text[:500],  # truncate to keep prompt short
                story_title=story_title,
                age=age,
                language=language,
            )
            # Use cheap model — generating an image prompt is a simple task
            image_prompt = await self.ai_service.generate_content(
                prompt,
                model_override=settings.GEMINI_MODEL,
            )
            return image_prompt.strip().replace("```", "")
        except Exception as e:
            logger.warning(f"[ImageGenerator] LLM image-prompt generation failed, using fallback: {e}")
            return (
                f"A colourful, child-friendly illustration for a children's story titled "
                f"'{story_title}'. Age group: {age}. Bright colours, simple shapes, "
                f"whimsical and safe for children."
            )

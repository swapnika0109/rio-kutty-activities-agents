"""
ImageGeneratorAgent — generates a cover image for a children's story.

Model: FLUX.1-schnell via HuggingFace InferenceClient
  - ~10x cheaper than FLUX.1-dev (4 inference steps vs 50), Apache-2.0 license
  - Quality is sufficient for children's story illustrations

Uses the image_prompt from the story creator response directly, appending
age-appropriate animated style suffixes before passing to FLUX.1-schnell.
"""

from ...services.ai_service import AIService
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageGeneratorAgent:
    def __init__(self):
        self.ai_service = AIService()

    async def generate(self, state: dict) -> dict:
        """
        Generates a story cover image.

        Expected state fields:
            story_text, story_title, age, language
            image_prompt (optional) — from story creator; used directly if present

        Returns partial state update with:
            state["image_bytes"] — raw PNG bytes (or None on failure)
            state["image_prompt"] — the prompt used for image generation (for evaluation)
        """
        story_text = state.get("story_text", "")
        story_title = state.get("story_title", "")
        age = state.get("age", "3-4")
        language = state.get("language", "English")
        base_prompt = state.get("image_prompt", "")

        # Step 1: Build final image prompt — use story's image_prompt if available,
        # else fall back to LLM generation. Then append age/style suffixes.
        image_prompt = self._build_image_prompt(base_prompt, story_title, age)

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

    def _build_image_prompt(self, base_prompt: str, story_title: str, age: str) -> str:
        """
        Takes the story's image_prompt and appends age-appropriate animated style suffixes.
        Falls back to a minimal prompt if base_prompt is empty.
        """
        base = base_prompt.strip() if base_prompt else (
            f"A children's book illustration for a story titled '{story_title}'"
        )
        style = (
            "3D animated movie style"
        )
        return f"{base}, {style}"

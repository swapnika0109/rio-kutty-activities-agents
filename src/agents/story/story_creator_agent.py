"""
StoryCreatorAgent — generates a full children's story from a selected topic.

Model: gemini-2.5-flash-preview-04-17 (higher quality; story generation benefits from
       a stronger model to produce coherent, engaging narratives)
Prompt: src/prompts/story_creator/v1.txt (user-supplied)
Output format expected from LLM:
    {
        "title": "...",
        "story_text": "...",
        "moral": "...",
        "age_group": "...",
        "language": "...",
        "character_names": [...],
        "setting": "..."
    }
"""

import json
from ...services.ai_service import AIService
from ...utils.logger import setup_logger
from ...utils.config import get_settings
from ...prompts import get_registry

logger = setup_logger(__name__)
settings = get_settings()

# Maps the language string from the API to the prompt file suffix
_LANG_CODE = {
    "English": "en",
    "Telugu":  "te",
}


class StoryCreatorAgent:
    def __init__(self, prompt_version: str = "1"):
        self.ai_service = AIService()
        # Version is just the number — the agent builds "v{N}_{lang}" itself
        self.prompt_version = prompt_version

    async def generate(self, state: dict) -> dict:
        """
        Generates a full story from the selected topic.

        Expected state fields:
            selected_topic: dict (from WF1 human selection)
            age, language (from config.configurable via unpack_config)

        Returns partial state update with state["story"] set.
        """
        topic = state.get("selected_topic", {})
        age = state.get("age", "3-4")
        language = state.get("language", "English")

        topic_title = topic.get("title", "")
        topic_theme = topic.get("theme", "")
        topic_moral = topic.get("moral", "")
        topic_description = topic.get("description", "")
        filter_type = topic.get("filter_type", "")
        filter_value = topic.get("filter_value", "")

        lang_code = _LANG_CODE.get(language, "en")

        # Derive theme-specific template variables from the topic's filter fields
        country = filter_value if filter_type == "country" else state.get("country", "India")
        religion = filter_value if filter_type == "religion" else state.get("religion", "universal_wisdom")

        registry = get_registry()
        prompt = registry.get_prompt(
            f"story_creator/{topic_theme}",  # e.g. "story_creator/theme1"
            version=f"v{self.prompt_version}_{lang_code}",  # e.g. "v1_en" or "v1_te"
            # Common variables
            age=age,
            name="Rio",            # default child name for batch generation
            topic=topic_title,
            code=lang_code,
            category="",           # present in prompt header comments only
            # Theme 1 / Theme 3
            country=country,
            # Theme 2
            religion=religion,
            source=filter_value,   # e.g. "hindu", "christian"
            story_seed=topic_description,
            # Legacy kwargs (ignored if not in template)
            topic_title=topic_title,
            topic_theme=topic_theme,
            topic_moral=topic_moral,
            topic_description=topic_description,
        )

        try:
            response = await self.ai_service.generate_content(
                prompt,
                model_override=settings.STORY_CREATOR_MODEL,
                fallback_override=settings.STORY_CREATOR_FALLBACK_MODEL,
            )
            story = self._parse_story(response)
            logger.info(f"[StoryCreator] Generated story: '{story.get('title', 'untitled')}'")
            return {
                "story": story,
                "validated": False,
                "evaluation": None,
                "completed": [],
                "errors": {},
            }
        except Exception as e:
            logger.error(f"[StoryCreator] Failed: {e}")
            return {"errors": {"story_creator": str(e)}}

    def _parse_story(self, response: str) -> dict:
        """Parse JSON dict from LLM response."""
        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                return json.loads(response[start : end + 1])
            raise ValueError(f"Could not parse story from response: {response[:200]}")

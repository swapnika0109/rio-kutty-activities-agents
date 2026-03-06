"""
AudioGeneratorAgent — generates MP3 narration for a children's story.

Uses Google Cloud Text-to-Speech via AudioService.
Single audio file per story in the story's requested language.
Voice and language are read from config (TTS_VOICE_NAME, TTS_LANGUAGE_CODE)
but can be overridden via the workflow config.
"""

from ...services.audio_service import AudioService
from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()


class AudioGeneratorAgent:
    def __init__(self):
        self.audio_service = AudioService()

    async def generate(self, state: dict) -> dict:
        """
        Generates audio narration from story text.

        Expected state fields:
            story_text: str — full story text to narrate
            language: str  — BCP-47 code (e.g. "en-US", "ta-IN", "hi-IN")
            voice: str     — TTS voice name (optional, falls back to config)

        Returns partial state update with:
            state["audio_bytes"] — raw MP3 bytes (or None on failure)
        """
        story_text = state.get("story_text", "")
        language = state.get("language", settings.TTS_LANGUAGE_CODE)
        voice = state.get("voice", settings.TTS_VOICE_NAME)

        if not story_text:
            logger.warning("[AudioGenerator] Empty story_text, skipping audio generation")
            return {"errors": {**state.get("errors", {}), "audio_generator": "story_text is empty"}}

        try:
            audio_bytes = await self.audio_service.synthesize_with_fallback(
                text=story_text,
                language_code=language,
                voice_name=voice,
            )
            if audio_bytes is None:
                logger.warning("[AudioGenerator] TTS returned None")
                return {
                    "audio_bytes": None,
                    "errors": {**state.get("errors", {}), "audio_generator": "TTS returned None"},
                }
            logger.info(f"[AudioGenerator] Generated audio: {len(audio_bytes)} bytes")
            return {
                "audio_bytes": audio_bytes,
                "validated": False,
                "evaluation": None,
            }
        except Exception as e:
            logger.error(f"[AudioGenerator] Failed: {e}")
            return {
                "audio_bytes": None,
                "errors": {**state.get("errors", {}), "audio_generator": str(e)},
            }

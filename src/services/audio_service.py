"""
AudioService — Google Cloud Text-to-Speech integration.

Wraps the google-cloud-texttospeech API with the existing resilience patterns
(circuit breaker + retry + rate limiting) consistent with AIService.

Why Google TTS?
- Native GCP integration (same project credentials as Firestore/Storage)
- Supports Tamil (ta-IN), Hindi (hi-IN), English (en-US) and many other languages
- Cost-effective for children's story narration (pay-per-character)
- Returns audio bytes (MP3) directly, no streaming required for background processing
"""

import asyncio
import io
from google.cloud import texttospeech

from .database.firestore_service import FirestoreService  # for credential pattern reference
from ..utils.config import get_settings
from ..utils.logger import setup_logger
from ..utils.resilience import circuit_breaker, retry_with_backoff, CircuitBreakerError

logger = setup_logger(__name__)
settings = get_settings()


class AudioService:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> texttospeech.TextToSpeechClient:
        """Lazy-initialize TTS client using Application Default Credentials."""
        if self._client is None:
            self._client = texttospeech.TextToSpeechClient()
        return self._client

    @circuit_breaker(
        name="google_tts",
        failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
    )
    @retry_with_backoff(
        max_retries=settings.MAX_RETRIES,
        base_delay=settings.RETRY_DELAY_SECONDS,
    )
    async def synthesize_speech(
        self,
        text: str,
        language_code: str = None,
        voice_name: str = None,
    ) -> bytes:
        """
        Synthesizes speech from text using Google Cloud TTS.

        Args:
            text: Story text to narrate.
            language_code: BCP-47 code (e.g. "en-US", "ta-IN"). Defaults to config.
            voice_name: TTS voice name (e.g. "en-US-Standard-A"). Defaults to config.

        Returns:
            MP3 audio bytes.

        Raises:
            CircuitBreakerError: If TTS circuit is open.
            Exception: On synthesis failure after retries.
        """
        lang = language_code or settings.TTS_LANGUAGE_CODE
        voice = voice_name or settings.TTS_VOICE_NAME

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=voice,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding[settings.TTS_AUDIO_ENCODING],
        )

        logger.info(f"[AudioService] Synthesizing speech: lang={lang} voice={voice} chars={len(text)}")

        # TTS client is synchronous; run in executor to keep async loop free
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            ),
        )
        logger.info("[AudioService] Speech synthesis completed")
        return response.audio_content

    async def synthesize_with_fallback(
        self,
        text: str,
        language_code: str = None,
        voice_name: str = None,
    ) -> bytes | None:
        """
        Synthesizes speech with graceful fallback (returns None instead of raising).
        Used by AudioGeneratorAgent where a missing audio is recoverable via retry.
        """
        try:
            return await self.synthesize_speech(text, language_code, voice_name)
        except CircuitBreakerError:
            logger.error("[AudioService] TTS circuit breaker OPEN — audio generation unavailable")
            return None
        except Exception as e:
            logger.error(f"[AudioService] TTS synthesis failed: {e}")
            return None

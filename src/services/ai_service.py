from ..utils.config import get_settings
from ..utils.logger import setup_logger
from google import genai
from google.genai import types
import hashlib
import os
import mimetypes
import uuid
from functools import lru_cache
import io
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="together",
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-dev",
)

settings = get_settings()
logger = setup_logger(__name__)

class AIService:
    def __init__(self):
        # Initialize the new Client from google-genai
        self._client = None
        # Ensure we use a model that supports image generation if requested
        # e.g., "gemini-2.0-flash-exp" or "gemini-2.5-flash-image"
        self.model_name = settings.GEMINI_MODEL 
        self.multimodal_model_name = settings.MULTIMODAL_MODEL

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        return self._client

    @lru_cache(maxsize=100)
    def _generate_cached(self, prompt_hash: str, prompt: str):
        """
        Internal method to cache AI text responses.
        """
        logger.info(f"Generating new content for hash: {prompt_hash[:8]}...")

        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4000, 
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                ),
            ],
            response_mime_type="application/json",
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generate_content_config
        )
        return response.text

    async def generate_content(self, prompt: str) -> str:
        """
        Public method to generate text content.
        """
        try:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            return self._generate_cached(prompt_hash, prompt)
        except Exception as e:
            logger.error(f"AI Generation failed: {str(e)}")
            raise e

    async def generate_multimodal_content(self, prompt: str) -> dict:
        """
        Generates both TEXT and IMAGES from a single prompt using the new SDK.
        Returns a dict with 'text' and 'images' (list of dictionaries with 'mime_type' and 'data').
        """
        logger.info(f"Generating multimodal content for: {prompt[:30]}...")
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000, 
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                    ),
                ],
                response_mime_type="application/json",
                response_modalities=["IMAGE", "TEXT"],
            )

            text_parts = []
            images = []
            
            # Using streaming to handle mixed content
            # Note: This is synchronous in the SDK currently, but wrapped in async method
            for chunk in self.client.models.generate_content_stream(
                model=self.multimodal_model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue

                part = chunk.candidates[0].content.parts[0]
                
                # Handle Image
                if part.inline_data and part.inline_data.data:
                    images.append({
                        "mime_type": part.inline_data.mime_type,
                        "data": part.inline_data.data # Raw binary data
                    })
                
                # Handle Text
                if part.text:
                    text_parts.append(part.text)

            return {
                "text": "".join(text_parts).strip(),
                "images": images
            }

        except Exception as e:
            logger.error(f"Multimodal Generation failed: {str(e)}")
            raise e

    async def generate_image(self, prompt: str):
        """
        Generates an image from a prompt using the Together API.
        """
        try:
            logger.info(f"Generating image for: {prompt[:30]}...")
            client = InferenceClient(
                provider="together",
                api_key=settings.HF_TOKEN,
            )

            # output is a PIL.Image object
            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev",
            )
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')   
            img_byte_arr = img_byte_arr.getvalue()
            return img_byte_arr 
        except Exception as e:
            logger.error(f"Image Generation failed: {str(e)}")
            raise e

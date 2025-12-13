import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class CreativeAgent:
    def __init__(self):
        self.ai_service = AIService()

    async def generate(self, state: dict):
        logger.info("Starting Creative activity generation...")
        summary = state.get("story_text", "")
        age = state.get("age", 5)
        
        prompt = f"""
        Create a creative activity (e.g., role-play, what-if scenario) for a {age}-year-old based on: "{summary}"
        
        Output strictly in JSON format:
        {{
            "title": "Activity Title",
            "instructions": "How to play",
            "questions_to_ask": ["Question 1", "Question 2"]
        }}
        """
        
        try:
            response = await self.ai_service.generate_content(prompt)
            # Response is a dict: {"text": "...", "images": [...]}
            
            cleaned_text = response["text"].replace("```json", "").replace("```", "").strip()
            activity_data = json.loads(cleaned_text)
            
            # If images were generated, attach them
            # We assume the prompt asked for 1 image which corresponds to the activity
            if response["images"]:
                # For now, just taking the first image data
                # Ideally, we should upload this to cloud storage and get a URL here
                # But per your current flow, we will pass the binary data or handle it in the Saver.
                # Let's store it in a separate 'images' key in state for the saver to handle.
                pass 

            return {
                "activities": {**state.get("activities", {}), "creative": activity_data},
                "images": {**state.get("images", {}), "creative": response["images"]},
                "completed": state.get("completed", []) + ["creative"]
            }
        except Exception as e:
            logger.error(f"Creative Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "creative": str(e)}}

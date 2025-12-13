import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MatchingAgent:
    def __init__(self):
        self.ai_service = AIService()

    async def generate(self, state: dict):
        logger.info("Starting Matching activity generation...")
        summary = state.get("story_summary", "")
        age = state.get("age", 5)
        
        prompt = f"""
        Create a matching game (3 pairs) for a {age}-year-old based on: "{summary}"
        
        Output strictly in JSON format:
        {{
            "title": "Match the items",
            "pairs": [
                {{"item_1": "Description A", "item_2": "Description B"}},
                {{"item_1": "Description C", "item_2": "Description D"}}
            ],
            "image": "The image of the matching game" 
        }}
        """
        
        try:
            response = await self.ai_service.generate_multimodal_content(prompt)
            
            cleaned_text = response["text"].replace("```json", "").replace("```", "").strip()
            activity_data = json.loads(cleaned_text)
            
            return {
                "activities": {**state.get("activities", {}), "matching": activity_data},
                "images": {**state.get("images", {}), "matching": response["images"]},
                "completed": state.get("completed", []) + ["matching"]
            }
        except Exception as e:
            logger.error(f"Matching Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "matching": str(e)}}
import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ArtAgent:
    def __init__(self):
        self.ai_service = AIService()

    async def generate(self, state: dict):
        logger.info("Starting Art activity generation...")
        summary = state.get("story_text", "")
        age = state.get("age", 5)
        
        prompt = f"""
        Create a simple art/drawing activity for a {age}-year-old based on: "{summary}"
        
        Output strictly in JSON format:
        {{
            "title": "Activity Title",
            "description": "Step-by-step instructions",
            "materials_needed": ["paper", "crayons"]
        }}
        """
        
        try:
            response = await self.ai_service.generate_content(prompt)
            
            cleaned_text = response.replace("```json", "").replace("```", "").strip()
            activity_data = json.loads(cleaned_text)
            
            return {
                "activities": {**state.get("activities", {}), "art": activity_data},
                # "images": {**state.get("images", {}), "art": response["images"]},
                "completed": state.get("completed", []) + ["art"]
            }
        except Exception as e:
            logger.error(f"Art Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "art": str(e)}}

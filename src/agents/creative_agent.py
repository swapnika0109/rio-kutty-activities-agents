import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class CreativeAgent:
    def __init__(self):
        self.ai_service = AIService()

    async def generate(self, state: dict):
        logger.info("Starting Creative activity generation...")
        summary = state.get("story_summary", "")
        age = state.get("age", 5)
        
        prompt = f"""
        Create a creative activity (e.g., role-play, what-if scenario) for a {age}-year-old based on: "{summary}"
        
        Output strictly in JSON format:
        {{
            "title": "Activity Title",
            "instructions": "How to play",
            "questions_to_ask": ["Question 1", "Question 2"],
            "image": "The image of the creative activity"
        }}
        """
        
        try:
            response_text = await self.ai_service.generate_multimodal_content(prompt)
            cleaned_text = response_text.replace("", "").replace("```", "").strip()
            activity_data = json.loads(cleaned_text)
            
            return {
                "activities": {**state.get("activities", {}), "creative": activity_data},
                "completed": state.get("completed", []) + ["creative_text"]
            }
        except Exception as e:
            logger.error(f"Creative Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "creative": str(e)}}#### 3. `src/agents/matching_agent.py`

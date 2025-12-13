import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MCQAgent:
    def __init__(self):
        self.ai_service = AIService()


    async def generate(self, state: dict):
        """
        Generates MCQs based on the story summary.
        Expected state: { "story_text": "...", "age": 5, ... }
        """
        logger.info("Starting MCQ generation...")
        summary = state.get("story_text", "")
        age = state.get("age", 5)
        prompt = f"""
         Create 3 multiple-choice questions for a {age}-year-old based on this story:
        "{summary}"
        
        Output strictly in JSON format:
        [
            {{"question": "...", "options": ["A", "B", "C"], "correct": "A"}}
        ]
        """

        try:
            response = await self.ai_service.generate_content(prompt)
            cleaned_text = response.replace("```json", "").replace("```", "").strip()
            mcq_data = json.loads(cleaned_text)

            return {
                "activity_type": "mcq",
                "activities": {**state.get("activities", {}), "mcq": mcq_data},
                "completed": state.get("completed", []) + ["mcq"]
            }
        except Exception as e:
            logger.error(f"MCQ generation failed: {str(e)}")
            return {
                "errors": {**state.get("errors", {}), "mcq": str(e)}
            }

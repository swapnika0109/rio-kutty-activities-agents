from ..services.ai_service import AIService
from ..utils.logger import setup_logger
import json

logger = setup_logger(__name__)

class ScienceAgent:

    def __init__(self):
        self.ai_service = AIService()
    
    async def generate(self, state: dict):
        """
        Generates Science based activities for the story.
        Expected state: { "Instructions": "...", "Images": "..." }
        """

        logger.info("Starting Science based activities generation...")
        story = state.get("story_text", "")
        age = state.get("age", 5)
        language = state.get("language", "English")

        prompt = f"""
        Create a science based activity on the story below for the kids aged {age} years old.
        Rules :
         - On top share what the activity teaches and how it is related to the story.
         - Then add the short instructions and needed items in a structure.
        "story : {story}"
        
        Output strictly in JSON format:
        [
            {{"What it Teaches": "...", "Instructions": "1. Gather : ..., 2. Introduction of the activity : ..., 3.Observations from the activity : ..., 4. Discuss about the activity and explain why :....", "Story Connection": "A"}}
        ]"""

        try:
            response = await self.ai_service.generate_content(prompt)
            cleaned_text = response.replace("```json", "").replace("```", "").strip()
            science_data = json.loads(cleaned_text)
            return {
                "activity_type": "science",
                "activities": {**state.get("activities", {}), "science": science_data},
                "completed": state.get("completed", []) + ["science"]
            }
        except Exception as e:
            logger.error(f"Science based activities generation failed: {str(e)}")
            return {
                "errors": {**state.get("errors", {}), "science": str(e)}
            }

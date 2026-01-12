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
        Generate a science based activity on the story below for the kids aged {age} years old.
        Rules :
         - Activity should be in a way that they have different things to do like cutting, pasting, playing etc.
         - On top share what the activity teaches and how it is related to the story.
         - Then add the eloberate step by step instructions and needed items in a structure.
         - Instructions should be in a way that kids can understand and follow easily.
         - All the instructions should be very detailed to finish the activity.
        "story : {story}"
        
        Output strictly in valid JSON format with no markdown code blocks. 
        Ensure that all strings are properly escaped, especially double quotes inside the text.
        [
            {{"What it Teaches": "...", "Instructions": "...", "Story Connection": "..."}}
        ]"""

        try:
            response = await self.ai_service.generate_content(prompt)
            
            # Robust JSON extraction
            start_index = response.find('[')
            end_index = response.rfind(']')
            if start_index != -1 and end_index != -1:
                cleaned_text = response[start_index:end_index+1]
            else:
                cleaned_text = response.replace("```json", "").replace("```", "").strip()

            science_data = json.loads(cleaned_text)
            image = await self.ai_service.generate_image("Strictly no description or instructions on the image   Activity : " + science_data[0].get("Instructions", ""))

            return {
                "activity_type": "science",
                "activities": {**state.get("activities", {}), "science": science_data},
                "images": {**state.get("images", {}), "science": image},
                "completed": state.get("completed", []) + ["science"]
            }
        except Exception as e:
            logger.error(f"Science based activities generation failed: {str(e)}")
            return {
                "errors": {**state.get("errors", {}), "science": str(e)}
            }

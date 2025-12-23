import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MoralAgent:
    def __init__(self):
        self.ai_service = AIService()

    async def generate(self, state: dict):
        logger.info("Starting Creative activity generation...")
        story = state.get("story_text", "")
        age = state.get("age", 5)
        
        prompt = f"""
        Create a moral based activity on the story below for the kids aged {age} years old.
        Rules :
            - On top share what the activity teaches and how it is related to the story.
            - Then add the short instructions and needed items in a structure.
            - Always generate activity to make the kids understand the ethical moral of the story
            - Generate at least 2 activities
            - Activity should be in a way that moral has to be understand but shouldn't necessarily use the same story theme.
            - Use the creativity to use different materials and different themes to make them understand the moral.
            - Activity doesn't have to relate with the entire story instead, it should just relate with moral.


        "story : {story}"
        Output strictly in JSON format:
        [
        {{"What it Teaches": "...", "Instructions": "1. Gather : ..., 2. Introduction of the activity : ..., 3.Observations from the activity : ..., 4. Discuss about the activity :....", "Story Connection": "A"}}
        ]
        """
        
        try:
            response = await self.ai_service.generate_content(prompt)
            # Response is a dict: {"text": "...", "images": [...]}
            
            cleaned_text = response.replace("```json", "").replace("```", "").strip()
            activity_data = json.loads(cleaned_text)
            
            # If images were generated, attach them
            # We assume the prompt asked for 1 image which corresponds to the activity
            # if response["images"]:
                # For now, just taking the first image data
                # Ideally, we should upload this to cloud storage and get a URL here
                # But per your current flow, we will pass the binary data or handle it in the Saver.
                # Let's store it in a separate 'images' key in state for the saver to handle.
                # pass 

            return {
                "activities": {**state.get("activities", {}), "moral": activity_data},
                # "images": {**state.get("images", {}), "creative": response["images"]},
                "completed": state.get("completed", []) + ["moral"]
            }
        except Exception as e:
            logger.error(f"Creative Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "moral": str(e)}}

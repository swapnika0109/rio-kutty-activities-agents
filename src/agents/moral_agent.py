import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MoralAgent:
    def __init__(self):
        self.ai_service = AIService()

    async def generate(self, state: dict):
        logger.info("Starting Moral activity generation...")
        story = state.get("story_text", "")
        age = state.get("age", 5)
        
        prompt = f"""
        Generate a moral based activity on the story below for the kids aged {age} years old.
        Rules :
            - Activity should be in a way that they have different things to do like cutting, pasting, playing etc
            - On top share what the activity teaches and how it is related to the story.
            - Then add the eloberate step by step instructions and needed items in a structure.
            - Instructions should be in a way that kids can understand and follow easily.
            - All the instructions should be very detailed to finish the activity.
            - Always generate activity to make the kids understand the ethical moral of the story
            - Generate at least 2 activities
            - Activity should be in a way that moral has to be understand but shouldn't necessarily use the same story theme.
            - Use the creativity to use different materials and different themes to make them understand the moral.
            - Activity doesn't have to relate with the entire story instead, it should just relate with moral.


        "story : {story}"
        Output strictly in valid JSON format with no markdown code blocks. 
        Ensure that all strings are properly escaped, especially double quotes inside the text.
        [
            {{"What it Teaches": "...", "Instructions": "...", "Story Connection": "..."}}
        ]
        """
        
        try:
            response = await self.ai_service.generate_content(prompt)
            # Response is a dict: {"text": "...", "images": [...]}
            
            # Robust JSON extraction
            start_index = response.find('[')
            end_index = response.rfind(']')
            if start_index != -1 and end_index != -1:
                cleaned_text = response[start_index:end_index+1]
            else:
                cleaned_text = response.replace("```json", "").replace("```", "").strip()

            activity_data = json.loads(cleaned_text)
            if len(activity_data) >= 2:
                activity1 = activity_data[0].get("Instructions", "")
                activity2 = activity_data[1].get("Instructions", "")
                # Assuming there are at least two activities if "}," is found
                image1 = await self.ai_service.generate_image("Strictly no description or instructions on the image   Activity : "+activity1)
                activity_data[0]["image"] = image1
                image2 = await self.ai_service.generate_image("Strictly no description or instructions on the image   Activity : "+activity2)
                activity_data[1]["image"] = image2
            else:
                image = await self.ai_service.generate_image("Strictly no description or instructions on the image   Activity : "+activity_data[0].get("Instructions", ""))
                activity_data[0]["image"] = image

            
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
                "completed": state.get("completed", []) + ["moral"]
            }
        except Exception as e:
            logger.error(f"Moral Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "moral": str(e)}}

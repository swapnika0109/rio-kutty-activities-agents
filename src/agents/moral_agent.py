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
        Context : You are a kids activity generator based on the provided story summary.	Objective : Generate an atleast 2 unique activities from the morals of the story for {age}-years-old.
            Thinking Process:
            1. Predefined Check : Treat yourself as a guardian of the kid and check whether the activity is easy to explain them
            2. Skill Check :  At {age} years old, what are the child's physical limitations? (e.g., can they use scissors? Yes, small kids safety one's). 
            3. Concept : Extract a couple of morals from the story and generate the activity out of the moral. It  doesn't have to relate with the story theme or line.
            4. Steps : Break the activity into 5 or 6 ultra-simple steps.
            5. Visualization: Describe exactly what the finished activity final output looks like (colors, textures, shapes) for an image generator.
            Output Format: Provide ONLY valid JSON.
                [{{
                    "title": "Creative Activity Name",
                    "age_appropriateness": "Explanation of why this fits a {age}-year-old",
                    "What it Teaches" : "Explain what the activity teaches"
                    "materials": ["item1", "item2"],
                    "Instructions": ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"],
                    "image_generation_prompt": "A high-quality, top-down photo of the finished craft: [detailed description based on the activity]"
                }}]
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
                image1 = await self.ai_service.generate_image(activity_data[0].get("image_generation_prompt", ""))
                activity_data[0]["image"] = image1
                image2 = await self.ai_service.generate_image(activity_data[1].get("image_generation_prompt", ""))
                activity_data[1]["image"] = image2
            else:
                image = await self.ai_service.generate_image(activity_data[0].get("image_generation_prompt", ""))
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

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
        language = state.get("language", "English" )

        prompt = f"""
        Context : You are a kids activity generator based on the provided story summary.	Objective : Generate an atleast 2 unique activities from the morals of the story for {age}-years-old.
            Thinking Process:
            1. Predefined Check : Treat yourself as a guardian of the kid and check whether the activity is easy to explain them
            2. Skill Check :  At {age} years old, what are the child's physical limitations? (e.g., can they use scissors? Yes, small kids safety one's). 
            3. Concept : Extract a science concept, Identify one physical law or natural phenomenon in the story (like Physical/Tactile, Observation-based) from the story and generate the activity out of the science. It  doesn't have to relate with the story theme or line.
            4. Steps : Break the activity into 5 or 6 ultra-simple steps.
            5. Language: Use Easy and simple daily routine {language} language for activity generation.
            6. Final Check: Before generating activity, evaluate the activity in terms of do-able and understandability for the {age}-years-old. 
            7. Visualization: Describe exactly what the finished activity final output looks like (colors, textures, shapes) for an image generator.
            Output Format: Provide ONLY valid JSON.
                [{{
                    "title": "Creative Activity Name",
                    "age_appropriateness": "Explanation of why this fits a {age}-year-old",
                    "What it Teaches" : "Explain what the activity teaches"
                    "materials": ["item1", "item2"],
                    "Instructions": ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"], in {language} language
                    "image_generation_prompt": "A high-quality, top-down photo of the finished craft: [detailed description based on the activity in English language]"
                }}]
        """

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
            image = await self.ai_service.generate_image(science_data[0].get("image_generation_prompt", ""))
            science_data[0]["image"] = image
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

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
        language = state.get("language", "English")
        
        prompt = f"""
        Context: You are an expert Early Childhood Educator and Art or craft Instructor.
        Task: Create a home-based art or craft activity for a {age}-year-old based on the story: "{summary}".

        Thinking Process (Follow these steps before outputting JSON):
        1. Skill Check: At {age} years old, what are the child's physical limitations? (e.g., can they use scissors? Yes, small kids safety one's). 
        2. Concept: Select one simple object from the story.
        3. Constraint: Do not suggest 'Drawing' or 'Coloring' unless it is combined with a physical 3D material (like leaves, cotton, or pasta)."
        4. Steps: Break the activity into 3 ultra-simple steps.
        5. Language: Use Easy and simple daily routine {language} language for activity generation.
        6. Final Check: Before generating activity, evaluate the activity in terms of do-able and understandability for the {age}-years-old. 
        7. Visualization: Describe exactly what the finished craft looks like (colors, textures, shapes) for an image generator. 

        Output Format: Provide ONLY valid JSON.
        {{
            "title": "Creative Activity Name",
            "age_appropriateness": "Explanation of why this fits a {age}-year-old",
            "materials": ["item1", "item2"],
            "steps": ["Step 1", "Step 2", "Step 3"] in {language} language,
            "image_generation_prompt": "A high-quality, top-down photo of the finished craft: [detailed description based on the activity in English language]"
        }}
        """
        
        try:
            response = await self.ai_service.generate_content(prompt)
            try:
                activity_data = json.loads(response)
            except json.JSONDecodeError:
                # Robust JSON extraction
                start_index = response.find('[')
                end_index = response.rfind(']')
                if start_index != -1 and end_index != -1:
                    cleaned_text = response[start_index:end_index+1]
                else:
                    cleaned_text = response.replace("```json", "").replace("```", "").strip()
                activity_data = json.loads(cleaned_text)
            
            image = await self.ai_service.generate_image(activity_data.get("image_generation_prompt", ""))
            activity_data["image"] = image
            return {
                "activities": {**state.get("activities", {}), "art": activity_data},
                "completed": state.get("completed", []) + ["art"]
            }
        except Exception as e:
            logger.error(f"Art Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "art": str(e)}}

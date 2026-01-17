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
        
        Output strictly in valid JSON format with no markdown code blocks.
        Ensure that all strings are properly escaped.
        [
            {{
                "title": "Activity Title",
                "description": "Step-by-step instructions",
                "materials_needed": ["paper", "crayons"],
                "Instructions": "..." 
            }}
        ]
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

            activity_data = json.loads(cleaned_text)
            # image = await self.ai_service.generate_image("Strictly no description or instructions on the image   Activity : " + activity_data[0].get("Instructions", ""))
            # activity_data[0]["image"] = image
            return {
                "activities": {**state.get("activities", {}), "art": activity_data},
                "completed": state.get("completed", []) + ["art"]
            }
        except Exception as e:
            logger.error(f"Art Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "art": str(e)}}

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
        context : you are a multi-agent activtity generator based on the story.
        agent 1 : context driver.
        agent 2: safety auditor.
        story : {story}
        objective :  Generate an unique science experiement or activity on story for {age}-years-old.
        Thinking Process:
        1) Predefined check :  As a context driver check the story carefully and treat yourself as guardian of the kid.
        2) Skill check : safety auditor has to check child’s pysical limitations.(e.g., can they use scissors? Yes, small kids safety one's).
       3) Concept : context driver is responsible to extract 
            - A Scientific Phenomenon (e.g., how water moves, gravity, or friction) and explain it as a 'Magic Trick' of nature by drawing conclusions based on evidence.
            - Attach the experiment with the charecters in the story.
        4) Evaluate : context driver has to evaluate the phenomenons that are extracted and pick the best one.
        5) Explaination : Safety auditor has to check the phenomenon is easy to explain or not to {age}-years-old.
        6) Remember : context driver should remember elements (like water, air, fire, earth, solids and liquids) can also be part of other stories, which has high chance of generating similar experiments.
        7) Generation: context driver should generate a unique experiment to this story, by adding new elements which makes the experiment unique.
        8) Testing experiment : Safety auditor should test the experiment and check whether
        it is suitable for {age}-years-old or not, in terms of understanding and safety(eg. if fire involves, say No).
        Also check the experiment is unique to this story.
        9) Steps : context driver Break the experiment into 5 to 6 ultrs-simple steps.
        10) Steps evalution :  Safety audito evalute each step against the safety of the kid.
        11) Language: context driver should use easy and simple daily routine {language} language for activity generation.
        12) Final Check: Safety auditor evaluate the experiment in terms of do-able, safety and understandability for the {age}-years-old  before generating. 
        13) Visualization: Describe exactly what the finished activity final output looks like (colors, textures, shapes) for an image generator.
        Output Format: Provide ONLY valid JSON.
            [{{
                "title": "Creative Activity Name",
                "age_appropriateness": "Explanation of why this fits a {age}-year-old",
                "What it Teaches" : "Explain the phenomenon of science it teaches in detail",
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

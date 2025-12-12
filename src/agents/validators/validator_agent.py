from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ValidatorAgent:
    def _increment_retry(self, state: dict, activity_type: str):
        current_retries = state.get("retry_count", {}).get(activity_type, 0)
        new_retry_count = {**state.get("retry_count", {}), activity_type: current_retries + 1}
        return {"retry_count": new_retry_count}

    def validate_mcq(self, state: dict):
        activities = state.get("activities", {})
        data = activities.get("mcq")
        
        if not data or not isinstance(data, list) or len(data) == 0:
            logger.warning("MCQ validation failed.")
            return self._increment_retry(state, "mcq")
            
        logger.info("MCQ validation passed.")
        return state

    def validate_art(self, state: dict):
        activities = state.get("activities", {})
        data = activities.get("art")
        
        # Check for required fields
        required = ["title", "description", "image_prompt"]
        if not data or not all(k in data for k in required):
            logger.warning("Art validation failed.")
            return self._increment_retry(state, "art")
            
        logger.info("Art validation passed.")
        return state

    def validate_creative(self, state: dict):
        activities = state.get("activities", {})
        data = activities.get("creative")
        
        required = ["title", "instructions", "image_prompt"]
        if not data or not all(k in data for k in required):
            logger.warning("Creative validation failed.")
            return self._increment_retry(state, "creative")
            
        logger.info("Creative validation passed.")
        return state

    def validate_matching(self, state: dict):
        activities = state.get("activities", {})
        data = activities.get("matching")
        
        if not data or "pairs" not in data or len(data["pairs"]) < 2:
            logger.warning("Matching validation failed.")
            return self._increment_retry(state, "matching")
            
        logger.info("Matching validation passed.")
        return state---
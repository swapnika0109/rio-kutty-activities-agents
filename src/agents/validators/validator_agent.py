from ...utils.logger import setup_logger

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
        return {
                "activities": {**state.get("activities", {}), "mcq": data},
                # "images": {**state.get("images", {}), "art": response["images"]},
               "completed": ["mcq"]
            }

    def validate_art(self, state: dict):
        activities = state.get("activities", {})
        data = activities.get("art")
        
        # Check for required fields
        required = ["title", "description"]
        if not data or not all(k in data for k in required):
            logger.warning("Art validation failed.")
            return self._increment_retry(state, "art")
            
        logger.info("Art validation passed.")
        return {
                "activities": {**state.get("activities", {}), "art": data},
                # "images": {**state.get("images", {}), "art": response["images"]},
               "completed": ["art"]
            }

    def validate_moral(self, state: dict):
        activities = state.get("activities", {})
        data = activities.get("moral")
        
        required = ["What it Teaches", "Instructions", "Story Connection", "image"]
        
        # Check if data is a non-empty list and its first element has all required fields
        is_valid = (
            isinstance(data, list) and 
            len(data) > 0 and 
            all(k in data[0] for k in required)
        )

        if not is_valid:
            logger.warning("Moral validation failed.")
            return self._increment_retry(state, "moral")
            
        logger.info("Moral validation passed.")
        return {
                "activities": {**state.get("activities", {}), "moral": data},
                "completed": ["moral"]
            }

    def validate_science(self, state: dict):
        activities = state.get("activities", {})
        data = activities.get("science")
        
        required = ["What it Teaches", "Instructions", "Story Connection", "image"]
        
        # Check if data is a non-empty list and its first element has all required fields
        is_valid = (
            isinstance(data, list) and 
            len(data) > 0 and 
            all(k in data[0] for k in required)
        )

        if not is_valid:
            logger.warning("Science validation failed.")
            return self._increment_retry(state, "science")
            
        logger.info("Science validation passed.")
        return {
                "activities": {**state.get("activities", {}), "science": data},
                "completed": ["science"]
            }
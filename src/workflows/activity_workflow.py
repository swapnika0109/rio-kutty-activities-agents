from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RunnableConfig

# Import Agents
from ..agents.mcq_agent import MCQAgent
from ..agents.art_agent import ArtAgent
from ..agents.creative_agent import CreativeAgent
from ..agents.matching_agent import MatchingAgent
from ..agents.validators.validator_agent import ValidatorAgent
from ..services.database.firestore_service import FirestoreService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# Reducer function to merge dictionaries
def merge_dicts(a: Dict, b: Dict) -> Dict:
    return {**a, **b}

# Update State Definition
class ActivityState(TypedDict):
    story_id: str
    story_text: str
    age: int
    language: str
    # Use Annotated to handle parallel updates
    activities: Annotated[Dict[str, Any], merge_dicts]
    images: Annotated[Dict[str, str], merge_dicts]
    completed: Annotated[List[str], operator.add]
    errors: Annotated[Dict[str, str], merge_dicts]
    retry_count: Annotated[Dict[str, int], merge_dicts]

# Initialize Components
# (Note: ImageService is now initialized INSIDE agents)
mcq_agent = MCQAgent()
art_agent = ArtAgent()
creative_agent = CreativeAgent()
matching_agent = MatchingAgent()
validator = ValidatorAgent()
firestore_service = FirestoreService()

def unpack_config(state: ActivityState, config: RunnableConfig):
    # Access read only data from config
    settings = config.get("configurable", {})
    story_id = settings.get("story_id")
    story_text = settings.get("story_text")
    age = settings.get("age")
    language = settings.get("language")
    return {**state, "story_id": story_id, "story_text": story_text, "age": age, "language": language}

# --- Generation Nodes ---
async def generate_mcq_node(state: ActivityState, config: RunnableConfig):
    return await mcq_agent.generate(unpack_config(state, config))

async def generate_art_node(state: ActivityState, config: RunnableConfig): return await art_agent.generate(unpack_config(state, config))
async def generate_creative_node(state: ActivityState, config: RunnableConfig): return await creative_agent.generate(unpack_config(state, config))
async def generate_matching_node(state: ActivityState, config: RunnableConfig): return await matching_agent.generate(unpack_config(state, config))

# --- Validation Nodes ---
def validate_mcq_node(state: ActivityState, config: RunnableConfig): return validator.validate_mcq(unpack_config(state, config))
def validate_art_node(state: ActivityState, config: RunnableConfig): return validator.validate_art(unpack_config(state, config))
def validate_creative_node(state: ActivityState, config: RunnableConfig): return validator.validate_creative(unpack_config(state, config))
def validate_matching_node(state: ActivityState, config: RunnableConfig): return validator.validate_matching(unpack_config(state, config))

# --- Save Nodes ---
async def save_mcq_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "mcq" in state.get("activities", {}):
        data = state["activities"]["mcq"]
        logger.info(f"Saving MCQ for story {db_data['story_id']}: {data}")
        await firestore_service.save_activity(db_data["story_id"], "mcq", db_data["activities"]["mcq"])
    return {}

async def save_art_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "art" in state.get("activities", {}):
        data = state["activities"]["art"]
        # data["image_url"] = db_data.get("images", {}).get("art")
        await firestore_service.save_activity(db_data["story_id"], "art", db_data["activities"]["art"])
    return {}

async def save_creative_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "creative" in state.get("activities", {}):
        data = state["activities"]["creative"]
        # data["image_url"] = db_data.get("images", {}).get("creative")
        await firestore_service.save_activity(db_data["story_id"], "creative", db_data["activities"]["creative"])
    return {}

async def save_matching_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "matching" in state.get("activities", {}):
        data = state["activities"]["matching"]
        # data["image_urls"] = db_data.get("images", {}).get("matching")
        await firestore_service.save_activity(db_data["story_id"], "matching", db_data["activities"]["matching"])
    return {}

# --- Routing Logic ---
def create_retry_logic(activity_type: str):
    def should_retry(state: ActivityState):
        if activity_type in state.get("errors", {}): return "fail"
        
        has_activity = activity_type in state.get("activities", {})
        retries = state.get("retry_count", {}).get(activity_type, 0)
        
        if has_activity and retries == 0: 
            return "next"
        if retries < 3:
            return "retry"
        return "fail"
    return should_retry
    

# --- Graph Construction ---
workflow = StateGraph(ActivityState)

# Add Nodes
workflow.add_node("start", lambda s: s) # Dummy start node

# # Activity 1: MCQ
workflow.add_node("gen_mcq", generate_mcq_node)
workflow.add_node("val_mcq", validate_mcq_node)
workflow.add_node("save_mcq", save_mcq_node)

# Activity 2: Art
workflow.add_node("gen_art", generate_art_node)
workflow.add_node("val_art", validate_art_node)
workflow.add_node("save_art", save_art_node)

# # Activity 3: Creative
workflow.add_node("gen_crt", generate_creative_node)
workflow.add_node("val_crt", validate_creative_node)
workflow.add_node("save_crt", save_creative_node)

# # Activity 4: Matching
workflow.add_node("gen_mat", generate_matching_node)
workflow.add_node("val_mat", validate_matching_node)
workflow.add_node("save_mat", save_matching_node)

# Entry & Fan-out
workflow.set_entry_point("start")
workflow.add_edge("start", "gen_mcq")
workflow.add_edge("start", "gen_art")
workflow.add_edge("start", "gen_crt")
workflow.add_edge("start", "gen_mat")

# Define Flows (Standardized: Gen -> Val -> Retry/Save)
for key, prefix in [("mcq", "mcq"), ("art", "art"), ("creative", "crt"), ("matching", "mat")]:
    gen, val, save = f"gen_{prefix}", f"val_{prefix}", f"save_{prefix}"
    
    workflow.add_edge(gen, val)
    workflow.add_conditional_edges(
        val, 
        create_retry_logic(key), 
        {"next": save, "retry": gen, "fail": END}
    )
    workflow.add_edge(save, END)

app_workflow = workflow.compile(checkpointer=MemorySaver())
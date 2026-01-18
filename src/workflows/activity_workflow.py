from typing import TypedDict, List, Dict, Any, Annotated
import operator
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import Agents
from ..agents.mcq_agent import MCQAgent
from ..agents.art_agent import ArtAgent
from ..agents.moral_agent import MoralAgent
from ..agents.science_agent import ScienceAgent
from ..agents.validators.validator_agent import ValidatorAgent
from ..services.database.firestore_service import FirestoreService
from ..services.database.storage_bucket import StorageBucketService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# Reducer function to merge dictionaries
def merge_dicts(a: Dict, b: Dict) -> Dict:
    return {**a, **b}

# Update State Definition
class ActivityState(TypedDict):
    # Removed story_id, story_text, age, language from state as they are now in config
    # Use Annotated to handle parallel updates
    activities: Annotated[Dict[str, Any], merge_dicts]
    images: Annotated[Dict[str, str], merge_dicts]
    completed: Annotated[List[str], operator.add]
    errors: Annotated[Dict[str, str], merge_dicts]
    retry_count: Annotated[Dict[str, int], merge_dicts]

# Initialize Components
mcq_agent = MCQAgent()
art_agent = ArtAgent()
moral_agent = MoralAgent()
science_agent = ScienceAgent()
validator = ValidatorAgent()
firestore_service = FirestoreService()
storage_bucket_service = StorageBucketService()


def unpack_config(state: ActivityState, config: RunnableConfig):
    # Access read only data from config
    settings = config.get("configurable", {})
    # Return a merged dict for the agents to use
    return {
        **state, 
        "story_id": settings.get("story_id"), 
        "story_text": settings.get("story_text"), 
        "age": settings.get("age"), 
        "language": settings.get("language")
    }

# --- Generation Nodes ---
async def generate_mcq_node(state: ActivityState, config: RunnableConfig):
    result = await mcq_agent.generate(unpack_config(state, config))
    current_retry = state.get("retry_count", {}).get("mcq", 0)
    result["retry_count"] = {**state.get("retry_count", {}), "mcq": current_retry + 1}
    return result

async def generate_art_node(state: ActivityState, config: RunnableConfig): 
    result = await art_agent.generate(unpack_config(state, config))
    current_retry = state.get("retry_count", {}).get("art", 0)
    result["retry_count"] = {**state.get("retry_count", {}), "art": current_retry + 1}
    return result

async def generate_moral_node(state: ActivityState, config: RunnableConfig): 
    result = await moral_agent.generate(unpack_config(state, config))
    current_retry = state.get("retry_count", {}).get("moral", 0)
    result["retry_count"] = {**state.get("retry_count", {}), "moral": current_retry + 1}
    return result

async def generate_science_node(state: ActivityState, config: RunnableConfig): 
    result = await science_agent.generate(unpack_config(state, config))
    current_retry = state.get("retry_count", {}).get("science", 0)
    result["retry_count"] = {**state.get("retry_count", {}), "science": current_retry + 1}
    return result

# --- Validation Nodes ---
def validate_mcq_node(state: ActivityState, config: RunnableConfig): 
    return validator.validate_mcq(unpack_config(state, config))

def validate_art_node(state: ActivityState, config: RunnableConfig): 
    return validator.validate_art(unpack_config(state, config))

def validate_science_node(state: ActivityState, config: RunnableConfig): 
    return validator.validate_science(unpack_config(state, config))

def validate_moral_node(state: ActivityState, config: RunnableConfig): 
    return validator.validate_moral(unpack_config(state, config))

# --- Save Nodes ---
async def save_mcq_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "mcq" in state.get("activities", {}):
        data = state["activities"]["mcq"]
        logger.info(f"Saving MCQ for story {db_data['story_id']}: {data}")
        await firestore_service.save_activity(db_data["story_id"], "mcq", data)
    return {}

async def save_art_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "art" in state.get("activities", {}):
        data = state["activities"]["art"][0]
        filename = await save_art_image_node(data, config)
        data["image"] = filename
        payload = {"items": data}
        await firestore_service.save_activity(db_data["story_id"], "art", payload)
    return {}

async def save_art_image_node(data: dict, config: RunnableConfig):
    if data.get("image", None):
        file_uuid = str(uuid.uuid4())
        filename = f"images/{file_uuid}.png"
        await storage_bucket_service.upload_file(filename, data.get("image", None))
        return filename
    return None

async def save_science_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "science" in state.get("activities", {}) :
        data = state["activities"]["science"][0]
        filename = await save_science_image_node(data, config)
        data["image"] = filename
        
        # Prepare payload for Firestore
        if isinstance(data, dict):
            payload = {**data}
        else:
            payload = {"items": data}
            
        await firestore_service.save_activity(db_data["story_id"], "science", payload)
    return {}
async def save_science_image_node(data: dict, config: RunnableConfig):
    if data.get("image", None):
        file_uuid = str(uuid.uuid4())
        filename = f"images/{file_uuid}.png"
        await storage_bucket_service.upload_file(filename, data.get("image", None))
        return filename
    return None

async def save_moral_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "moral" in state.get("activities", {}):
        data_list = state["activities"]["moral"]
        payloads = []
        for data in data_list:
            filename = await save_moral_image_node(data, config)
            data["image"] = filename
            # Prepare payload for Firestore
            if isinstance(data, dict):
                payload = {**data}
            else:
                payload = {"items": data}
            payloads.append(payload)
            
            
        await firestore_service.save_activity(db_data["story_id"], "moral", payloads)
    return {}
async def save_moral_image_node(data: dict, config: RunnableConfig):
    if data.get("image", None):
        file_uuid = str(uuid.uuid4())
        filename = f"images/{file_uuid}.png"
        await storage_bucket_service.upload_file(filename, data.get("image", None))
        return filename
    return None

# --- Routing Logic ---
def create_retry_logic(activity_type: str):
    def should_retry(state: ActivityState):
        if activity_type in state.get("errors", {}): return "fail"
        
        has_activity = activity_type in state.get("activities", {})
        retries = state.get("retry_count", {}).get(activity_type, 0)
        
        # If we have the activity
        if has_activity: 
            return "next"
            
        if retries < 3:
            return "retry"
        return "fail"
    return should_retry

# Helper to Check Exists & Route (Runs AT RUNTIME)
async def route_start(state: ActivityState, config: RunnableConfig):
    story_id = config["configurable"]["story_id"]
    nodes_to_run = []
    
    # Map activity types to their node name prefixes
    type_to_prefix = {
        "mcq": "mcq",
        "art": "art",
        "moral": "mor",
        "science": "sci"
    }
    
    for activity_type, prefix in type_to_prefix.items():
        # Check if it exists in DB
        exists = await firestore_service.check_if_activity_exists(story_id, activity_type)
        
        if not exists:
            # If NOT exists, we want to run the generator
            nodes_to_run.append(f"gen_{prefix}")
        else:
            logger.info(f"Skipping {activity_type} for {story_id} - already exists.")
            
    return nodes_to_run

# --- Graph Construction ---
workflow = StateGraph(ActivityState)

# Add Nodes
workflow.add_node("start", lambda s: s) # Dummy start node

# Activity 1: MCQ
workflow.add_node("gen_mcq", generate_mcq_node)
workflow.add_node("val_mcq", validate_mcq_node)
workflow.add_node("save_mcq", save_mcq_node)

# Activity 2: Art
workflow.add_node("gen_art", generate_art_node)
workflow.add_node("val_art", validate_art_node)
workflow.add_node("save_art", save_art_node)

# Activity 3: Creative
workflow.add_node("gen_mor", generate_moral_node)
workflow.add_node("val_mor", validate_moral_node)
workflow.add_node("save_mor", save_moral_node)

# Activity 4: Science
workflow.add_node("gen_sci", generate_science_node)
workflow.add_node("val_sci", validate_science_node)
workflow.add_node("save_sci", save_science_node)

# Entry & Fan-out (Dynamic)
workflow.set_entry_point("start")
workflow.add_conditional_edges(
    "start", 
    route_start,
    # Define possible destinations
    ["gen_mcq", "gen_art", "gen_mor", "gen_sci"] 
)

# Define Flows (Standardized: Gen -> Val -> Retry/Save)
for key, prefix in [("mcq", "mcq"), ("art", "art"), ("moral", "mor"), ("science", "sci")]:
    gen, val, save = f"gen_{prefix}", f"val_{prefix}", f"save_{prefix}"
    
    workflow.add_edge(gen, val)
    workflow.add_conditional_edges(
        val, 
        create_retry_logic(key), 
        {"next": save, "retry": gen, "fail": END}
    )
    workflow.add_edge(save, END)

app_workflow = workflow.compile(checkpointer=MemorySaver())
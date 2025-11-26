from enum import Enum

# Define the types of learning tasks available
class TaskType(str, Enum):
    """Defines the available learning tasks."""
    IMAGE_LABELING = "image_labeling"
    CONVERSATIONAL = "conversational_roleplay"
    GRAMMAR_DRILL = "grammar_drill"
    SENTENCE_REWRITE = "sentence_rewrite"

# Define the task request structure (Pydantic schema)
class TaskRequest(BaseModel):
    user_id: str
    task_type: TaskType
    # Add generalization parameters
    difficulty_level: str = "B1"
    num_items: int = 5
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict

# --- A. ENUMS (Fixed Choices) ---

class TaskType(str, Enum):
    """Defines the available learning tasks (Image/Conversation/Drill)."""
    IMAGE_LABELING = "image_labeling"
    CONVERSATIONAL = "conversational_roleplay"
    GRAMMAR_DRILL = "grammar_drill"
    SENTENCE_REWRITE = "sentence_rewrite"
    VOCABULARY_GENERATION = "vocabulary_generation" # <-- New Task Type for your friend's app logic

# --- B. REQUEST SCHEMA (What the Robot Sends) ---

class TaskRequest(BaseModel):
    """Schema for requesting a new learning task."""
    user_id: str
    task_type: TaskType
    # Generalized parameters for adaptive learning:
    difficulty_level: str = "A1" # Used to retrieve vocabulary sets (e.g., from a DB)
    num_items: int = 5          # Used for quantity in IMAGE_LABELING or VOCAB_GEN
    topic: str = "general"      # Used to guide the LLM (e.g., 'food', 'travel')

# --- C. TASK ITEM SCHEMA (The Core LLM Output) ---

class VocabItem(BaseModel):
    """Defines the structure for a single generated vocabulary word."""
    german: str
    english: str
    # Add a field for the LLM's confidence or source PDF page if needed later
    # source_confidence: float = 1.0 

class ImageVocabItem(BaseModel):  # <-- THIS NAME MUST EXIST EXACTLY AS WRITTEN
    """Defines the structure for a single image and its label."""
    image_url: str
    german_label: str
    english_translation: str
    
# --- D. RESPONSE SCHEMA (What the Backend Sends Back) ---

class TaskResponse(BaseModel):
    """The structure of the final payload sent to the robot to start the task."""
    session_id: str
    task_type: TaskType
    
    # Placeholder for the actual task content (list of words, scenario text, etc.)
    # The 'payload' field will contain the list of VocabItems for the VOCAB_GEN task.
    payload: Dict
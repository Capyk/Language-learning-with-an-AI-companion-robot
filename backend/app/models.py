from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# --- ENUMS ---
class TaskType(str, Enum):
    IMAGE_LABELING = "image_labeling"
    CONVERSATIONAL = "conversational_roleplay"
    GRAMMAR_DRILL = "grammar_drill"
    SENTENCE_REWRITE = "sentence_rewrite"
    VOCABULARY_GENERATION = "vocabulary_generation"
    GEC_CHALLENGE = "gec_challenge"
    LEARNING_STEP = "learning_step"

# --- REQUEST SCHEMAS ---
class TaskRequest(BaseModel):
    user_id: str
    task_type: TaskType
    difficulty_level: str = "A1"
    num_items: int = 5
    topic: str = "general"

class SessionInit(BaseModel):
    user_id: str
    condition: str
    access_code: Optional[str] = None # Opcjonalne w tej fazie

class AnswerSubmit(BaseModel):
    session_id: str
    user_answer: str
    start_time: float
    history: Optional[List[str]] = []

class SkipPhaseRequest(BaseModel):
    session_id: str
    phase: str 

# --- LEARNING SCREEN SCHEMA ---
class LearningScreen(BaseModel):
    step_number: int
    title: str
    content: str
    visual_type: str  # 'word_card', 'story', 'summary', 'challenge', 'intro'
    
    german_word: Optional[str] = None
    article: Optional[str] = None 
    plural: Optional[str] = None
    image_url: Optional[str] = None
    example_sentence: Optional[str] = None
    mnemonics: Optional[str] = None
    
    interaction_type: Optional[str] = "read_only"
    question_context: Optional[str] = None
    options: Optional[List[str]] = None

# --- RESPONSE SCHEMAS ---
class VocabItem(BaseModel):
    german: str
    english: str

class ImageVocabItem(BaseModel):
    image_id: str
    image_url: str
    german_word: str
    english_gloss: str
    article: Optional[str] = None
    plural: Optional[str] = None

class DemographicData(BaseModel):
    session_id: str
    age: Optional[str] = None
    gender: str
    education: str
    german_level: str
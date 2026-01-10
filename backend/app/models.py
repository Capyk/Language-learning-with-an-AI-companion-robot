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
    access_code: Optional[str] = None 

class AnswerSubmit(BaseModel):
    session_id: str
    user_answer: str
    start_time: float
    history: Optional[List[str]] = []

class SkipPhaseRequest(BaseModel):
    session_id: str
    phase: str 

# --- TUTOR SCHEMAS ---
class TutorTaskContext(BaseModel):
    prompt: str
    user_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    exercise_type: Optional[str] = None

class TutorRequest(BaseModel):
    session_id: str
    question: str
    task_context: TutorTaskContext
    response_language: str = "de"  # 'de' for German, 'en' for English

# --- LEARNING SCREEN SCHEMA ---
class LearningScreen(BaseModel):
    step_number: int
    title: str
    content: str
    visual_type: str  
    
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
    age: str
    gender: str
    education: str
    german_level: str
    questionnaire: Dict[str, Any] 

# Logika wyniku
class LogEntry(BaseModel):
    phase: str
    word: str
    task_type: str
    user_input: str
    # Score as float (1.0, 0.5, 0.0)
    score: float 
    is_correct: bool 
    timestamp: str
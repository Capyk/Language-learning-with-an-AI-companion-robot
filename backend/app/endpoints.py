import pandas as pd
import random
from datetime import datetime
from uuid import uuid4
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException

# Import internal modules
from .models import SessionInit, AnswerSubmit
from .llm_service import real_time_correction

router = APIRouter()

# --- DATA INGESTION ---
try:
    VOCAB_DF = pd.read_csv("vocab.csv", sep=";")
    VOCAB_DF = VOCAB_DF.dropna(subset=['image_id', 'german_word'])
except Exception as e:
    print(f"Error loading vocab.csv: {e}")
    VOCAB_DF = pd.DataFrame()

# In-memory session cache
EXP_CACHE: Dict[str, dict] = {}

# --- UTILITY FUNCTIONS ---

def prepare_experiment_items():
    """Selects 20 unique words (4 per scenario) and sets items for Pre/Post tests."""
    scenarios = ["apartment_request", "travel", "swimming", "pet_sitting", "birthday"]
    learning_pool = []
    
    for sc in scenarios:
        cat_items = VOCAB_DF[VOCAB_DF['scenario'] == sc].to_dict('records')
        if len(cat_items) >= 4:
            learning_pool.extend(random.sample(cat_items, 4))
        else:
            learning_pool.extend(cat_items)

    test_items = random.sample(learning_pool, 5)
    
    task_types = ["article_mcq", "article_mcq", "plural_mcq", "plural_mcq", "type_word"]
    random.shuffle(task_types)
    
    pre_items = [item.copy() for item in test_items]
    post_items = [item.copy() for item in test_items]
    
    for i in range(len(pre_items)):
        pre_items[i]['assigned_task'] = task_types[i]
        post_items[i]['assigned_task'] = task_types[i]
        
    return learning_pool, pre_items, post_items

# --- ENDPOINTS ---

@router.post("/experiment/init")
async def init_experiment(data: SessionInit):
    session_id = str(uuid4())
    learn_items, pre_items, post_items = prepare_experiment_items()
    
    EXP_CACHE[session_id] = {
        "user_id": data.user_id,
        "condition": data.condition,
        "phase": "pre-test",
        "current_index": 0,
        "attempt_count": 0,
        "items": {
            "pre-test": pre_items,
            "learning": learn_items,
            "post-test": post_items
        },
        "logs": []
    }
    
    return {"session_id": session_id, "condition": data.condition}

@router.get("/experiment/trial/{session_id}")
async def get_trial(session_id: str):
    session = EXP_CACHE.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    phase = session["phase"]
    idx = session["current_index"]
    
    if idx >= len(session["items"][phase]):
        if phase == "pre-test":
            session["phase"] = "learning"
            session["current_index"] = 0
        elif phase == "learning":
            session["phase"] = "post-test"
            session["current_index"] = 0
        else:
            return {"status": "completed"}
        
        phase = session["phase"]
        idx = 0

    item = session["items"][phase][idx]
    task_type = item.get('assigned_task', 'type_word')
    
    options = []
    if task_type == "article_mcq":
        options = ["der", "die", "das"]
    elif task_type == "plural_mcq":
        # FIXED: Distractors are now full words (noun + suffix) to match correct answer format
        correct = str(item['plural'])
        base = str(item['german_word'])
        
        # Possible plural patterns to generate distractors
        suffixes = ["en", "er", "e", "s", "n"]
        potential_distractors = [f"{base}{s}" for s in suffixes]
        
        # Filter out the correct one and select 2 random ones
        filtered_distractors = [d for d in potential_distractors if d.lower() != correct.lower()]
        options = list(set([correct] + random.sample(filtered_distractors, 2)))
        random.shuffle(options)

    image_url = f"/images/{item['image_id']}.jpg"

    return {
        "phase": phase,
        "index": idx,
        "total_in_phase": len(session["items"][phase]),
        "task_type": task_type,
        "image_url": image_url,
        "english_gloss": item['english_gloss'],
        "options": options if options else None,
        "german_word": item['german_word'] if task_type != "type_word" else None
    }

@router.post("/experiment/submit")
async def submit_answer(data: AnswerSubmit):
    session = EXP_CACHE.get(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    phase = session["phase"]
    idx = session["current_index"]
    item = session["items"][phase][idx]
    condition = session["condition"]
    
    task_type = item.get('assigned_task', 'type_word')
    if task_type == "article_mcq":
        correct_val = item['article']
    elif task_type == "plural_mcq":
        correct_val = item['plural']
    else:
        correct_val = f"{item['article']} {item['german_word']}"

    is_correct = str(data.user_answer).strip() == str(correct_val).strip()
    
    response_time = datetime.now().timestamp() - data.start_time
    session["logs"].append({
        "phase": phase,
        "item_id": item['image_id'],
        "task_type": task_type,
        "is_correct": is_correct,
        "response_time": response_time,
        "attempt": session["attempt_count"] + 1
    })

    if phase != "learning":
        session["current_index"] += 1
        return {"feedback": None, "is_correct": is_correct, "move_next": True}

    if is_correct:
        session["current_index"] += 1
        session["attempt_count"] = 0
        return {
            "is_correct": True,
            "feedback": f"Perfect! It is '{item['article']} {item['german_word']}'.",
            "example": item['example_de'],
            "move_next": True
        }
    else:
        session["attempt_count"] += 1
        
        if condition == "A":
            session["current_index"] += 1
            session["attempt_count"] = 0
            return {
                "is_correct": False,
                "feedback": f"Incorrect. The correct answer is: {item['article']} {item['german_word']}.",
                "move_next": True
            }
        else:
            if session["attempt_count"] >= 3:
                session["current_index"] += 1
                session["attempt_count"] = 0
                return {
                    "is_correct": False,
                    # FIXED: Neutral label for revealed answer
                    "feedback": f"The solution is: {item['article']} {item['german_word']}. {item['example_de']}",
                    "move_next": True
                }
            else:
                hint_data = await real_time_correction(
                    user_input=data.user_answer,
                    expected_answer=str(correct_val),
                    attempt_number=session["attempt_count"],
                    difficulty="A1"
                )
                return {
                    "is_correct": False,
                    "feedback": hint_data["tip"],
                    "move_next": False
                }

@router.get("/experiment/export/{session_id}")
async def export_data(session_id: str):
    session = EXP_CACHE.get(session_id)
    if not session: raise HTTPException(status_code=404)
    return {
        "user_id": session["user_id"],
        "condition": session["condition"],
        "logs": session["logs"]
    }
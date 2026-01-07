import pandas as pd
import random
from uuid import uuid4
from typing import Dict, List, Set
from fastapi import APIRouter, HTTPException
from datetime import datetime

from .models import SessionInit, AnswerSubmit, SkipPhaseRequest, LearningScreen, DemographicData
from .llm_service import generate_static_learning_path_A, generate_adaptive_learning_path_B
from .storage import init_db, save_session_result

router = APIRouter()

try:
    init_db()
except Exception:
    pass

# --- DATA LOADING ---
try:
    VOCAB_DF = pd.read_csv("vocab.csv", sep=";")
    VOCAB_DF = VOCAB_DF.dropna(subset=['image_id', 'german_word'])
    # Replace NaN with empty string to prevent JSON errors
    VOCAB_DF = VOCAB_DF.where(pd.notnull(VOCAB_DF), "")
except Exception as e:
    print(f"Error loading CSV: {e}")
    VOCAB_DF = pd.DataFrame()

EXP_CACHE: Dict[str, dict] = {}

def get_valid_variations(article_str: str, word_str: str) -> Set[str]:
    """Generates valid variations for slashed words (e.g., der Besitzer / die Besitzerin)."""
    valid_answers = set()
    articles = [a.strip() for a in str(article_str).split('/')]
    words = [w.strip() for w in str(word_str).split('/')]
    
    if len(articles) == len(words) and len(words) > 1:
        for i in range(len(words)):
            valid_answers.add(f"{articles[i]} {words[i]}".lower())
    else:
        for w in words:
            valid_answers.add(w.lower())
            for a in articles:
                valid_answers.add(f"{a} {w}".lower())
    return valid_answers

def prepare_experiment_items():
    scenarios = ["apartment_request", "travel", "swimming", "pet_sitting", "birthday"]
    learning_pool = []
    
    for sc in scenarios:
        cat_items = VOCAB_DF[VOCAB_DF['scenario'] == sc].to_dict('records')
        if len(cat_items) >= 4:
            learning_pool.extend(random.sample(cat_items, 4))
        else:
            learning_pool.extend(cat_items)    
            
    test_items = random.sample(learning_pool, 5)
    pre_items = [item.copy() for item in test_items]
    post_items = [item.copy() for item in test_items]
    
    # 3 Type Word tasks, 2 Plural MCQ tasks
    task_types = []
    for _ in range(3): task_types.append("type_word")
    for _ in range(2):
        if random.random() < 0.5: task_types.append("plural_mcq")
        else: task_types.append("type_word")
    random.shuffle(task_types)
    
    for i in range(len(pre_items)):
        task = task_types[i]
        # Safety: Don't show plural task if data is missing
        if task == "plural_mcq" and not str(pre_items[i].get('plural', '')).strip():
             task = "type_word"
             
        pre_items[i]['assigned_task'] = task
        post_items[i]['assigned_task'] = task
        
    return test_items, pre_items, post_items

# --- ENDPOINTS ---

@router.post("/experiment/init")
async def init_experiment(data: SessionInit):
    session_id = str(uuid4())
    learn_items, pre_items, post_items = prepare_experiment_items()
    
    session_data = {
        "session_id": session_id,
        "user_id": data.user_id,
        "condition": data.condition,
        "phase": "pre-test",
        "current_index": 0,
        "items": {
            "pre-test": pre_items,
            "learning_raw": learn_items,
            "learning_path": [],
            "post-test": post_items
        },
        "logs": []
    }
    
    if data.condition == "A":
        session_data["items"]["learning_path"] = generate_static_learning_path_A(learn_items)

    EXP_CACHE[session_id] = session_data
    return {"session_id": session_id, "condition": data.condition}

@router.post("/experiment/finalize")
async def finalize_experiment(data: DemographicData):
    session = EXP_CACHE.get(data.session_id)
    if not session: raise HTTPException(status_code=404)
    save_session_result(session, data.dict())
    return {"status": "success"}

@router.get("/experiment/trial/{session_id}")
async def get_trial(session_id: str):
    session = EXP_CACHE.get(session_id)
    if not session: raise HTTPException(status_code=404)
    phase = session["phase"]
    idx = session["current_index"]

    if phase == "learning":
        path = session["items"]["learning_path"]
        if idx >= len(path):
            session["phase"] = "post-test"
            session["current_index"] = 0
            return await get_trial(session_id)
        return {
            "phase": "learning",
            "index": idx,
            "total_in_phase": len(path),
            "task_type": "learning_step",
            "payload": path[idx]
        }

    items_list = session["items"][phase]
    if idx >= len(items_list):
        if phase == "post-test": return {"status": "completed"}
        return {"status": "transition"}

    item = items_list[idx]
    options = None
    
    if item.get('assigned_task') == 'plural_mcq':
        # Correct answer (clean)
        correct = str(item.get('plural', '')).split('/')[0].strip()
        
        # Distractors from CSV
        wrong1 = str(item.get('plural_wrong_1', '')).strip()
        wrong2 = str(item.get('plural_wrong_2', '')).strip()
        
        # Fallbacks just in case
        base = str(item.get('german_word', '')).split('/')[0]
        if not wrong1: wrong1 = f"{base}en"
        if not wrong2: wrong2 = f"{base}s"
            
        opts = [correct, wrong1, wrong2]
        opts = list(set(opts)) # unique
        # Pad if needed
        while len(opts) < 3:
             opts.append(f"{base}e" if f"{base}e" not in opts else f"{base}er")
             
        random.shuffle(opts)
        options = opts

    return {
        "phase": phase,
        "index": idx,
        "total_in_phase": len(items_list),
        "task_type": item.get('assigned_task', 'type_word'),
        "image_url": f"/images/{item['image_id']}.jpg",
        "english_gloss": item['english_gloss'],
        "options": options
    }

@router.post("/experiment/submit")
async def submit_answer(data: AnswerSubmit):
    session = EXP_CACHE.get(data.session_id)
    if not session: raise HTTPException(status_code=404)
    phase = session["phase"]
    
    # 1. Learning Phase Nav
    if phase == "learning":
        session["current_index"] += 1
        return {"is_correct": True, "move_next": True}

    # 2. Test Phase Logic
    items = session["items"][phase]
    if session["current_index"] >= len(items):
        return {"is_correct": False, "move_next": False, "transition": True}

    item = items[session["current_index"]]
    
    is_correct = False
    correct_display = ""
    
    if item.get('assigned_task') == 'plural_mcq':
        valid_plurals = [p.strip().lower() for p in str(item.get('plural', '')).split('/')]
        if data.user_answer.strip().lower() in valid_plurals:
            is_correct = True
        correct_display = item.get('plural', '').split('/')[0]
        
    elif item.get('assigned_task') == 'type_word':
        valid_variations = get_valid_variations(item.get('article', ''), item.get('german_word', ''))
        if data.user_answer.strip().lower() in valid_variations:
            is_correct = True
        
        s_art = str(item.get('article', '')).split('/')[0]
        s_word = str(item.get('german_word', '')).split('/')[0]
        correct_display = f"{s_art} {s_word}"

    # Log
    log_entry = {
        "phase": phase,
        "word": item.get('german_word'),
        "task_type": item.get('assigned_task'),
        "user_input": data.user_answer,
        "is_correct": is_correct,
        "timestamp": str(datetime.now())
    }
    session["logs"].append(log_entry)
    session["current_index"] += 1
    
    # Transition Logic
    if phase == "pre-test" and session["current_index"] >= len(items):
        if session["condition"] == "B":
            print(">>> Generowanie ścieżki AI...")
            path_b = await generate_adaptive_learning_path_B(session["items"]["learning_raw"], session["logs"])
            session["items"]["learning_path"] = path_b
        elif session["condition"] == "A" and not session["items"]["learning_path"]:
             session["items"]["learning_path"] = generate_static_learning_path_A(session["items"]["learning_raw"])
            
        session["phase"] = "learning"
        session["current_index"] = 0
        return {"is_correct": is_correct, "move_next": True, "transition": "learning", "feedback": f"Correct: {correct_display}"}

    return {"is_correct": is_correct, "move_next": True, "feedback": f"Correct: {correct_display}"}

@router.post("/experiment/skip")
async def skip_to_phase(data: SkipPhaseRequest):
    session = EXP_CACHE.get(data.session_id)
    if session:
        session["phase"] = data.phase
        session["current_index"] = 0
        if data.phase == "learning" and not session["items"]["learning_path"]:
             session["items"]["learning_path"] = generate_static_learning_path_A(session["items"]["learning_raw"])
    return {"status": "ok"}
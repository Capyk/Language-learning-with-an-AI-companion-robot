import pandas as pd
import random
from uuid import uuid4
from typing import Dict, List
from fastapi import APIRouter, HTTPException

from .models import SessionInit, AnswerSubmit, SkipPhaseRequest, LearningScreen
from .llm_service import generate_static_learning_path_A, generate_adaptive_learning_path_B, real_time_correction

router = APIRouter()

# --- DATA LOADING ---
try:
    VOCAB_DF = pd.read_csv("vocab.csv", sep=";")
    VOCAB_DF = VOCAB_DF.dropna(subset=['image_id', 'german_word'])
except Exception as e:
    print(f"Error loading CSV: {e}")
    VOCAB_DF = pd.DataFrame()

EXP_CACHE: Dict[str, dict] = {}

def prepare_experiment_items():
    """
    Losuje 5 unikalnych słów i przydziela typy zadań według nowej strategii:
    - 3 zadania to na pewno 'type_word' (wpisywanie).
    - 2 zadania to 50/50 szansa na 'plural_mcq' lub 'type_word'.
    - Brak zadania 'article_mcq'.
    """
    scenarios = ["apartment_request", "travel", "swimming", "pet_sitting", "birthday"]
    learning_pool = []
    
    # Pobieramy słowa z każdej kategorii
    for sc in scenarios:
        cat_items = VOCAB_DF[VOCAB_DF['scenario'] == sc].to_dict('records')
        if len(cat_items) >= 4:
            learning_pool.extend(random.sample(cat_items, 4))
        else:
            learning_pool.extend(cat_items)    
            
    # Wybieramy 5 słów do sesji
    test_items = random.sample(learning_pool, 5)
    
    pre_items = [item.copy() for item in test_items]
    post_items = [item.copy() for item in test_items]
    
    # --- NOWA LOGIKA PRZYDZIELANIA ZADAŃ ---
    task_types = []
    
    # 1. Trzy gwarantowane zadania na wpisywanie
    for _ in range(3):
        task_types.append("type_word")
        
    # 2. Dwa zadania losowane (50% Plural MCQ / 50% Wpisywanie)
    for _ in range(2):
        if random.random() < 0.5:
            task_types.append("plural_mcq")
        else:
            task_types.append("type_word")
            
    # 3. Mieszamy kolejność typów zadań, żeby nie było zawsze tak samo
    random.shuffle(task_types)
    
    # Przypisujemy te same typy zadań do pre-testu i post-testu dla spójności
    for i in range(len(pre_items)):
        pre_items[i]['assigned_task'] = task_types[i]
        post_items[i]['assigned_task'] = task_types[i]
        
    return test_items, pre_items, post_items

# --- ENDPOINTS ---

@router.post("/experiment/init")
async def init_experiment(data: SessionInit):
    session_id = str(uuid4())
    learn_items, pre_items, post_items = prepare_experiment_items()
    
    session_data = {
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

@router.get("/experiment/trial/{session_id}")
async def get_trial(session_id: str):
    session = EXP_CACHE.get(session_id)
    if not session: raise HTTPException(status_code=404)
    phase = session["phase"]
    idx = session["current_index"]

    # Faza Nauki
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

    # Pre/Post Test
    items_list = session["items"][phase]
    if idx >= len(items_list):
        if phase == "post-test": return {"status": "completed"}
        return {"status": "transition"}

    item = items_list[idx]
    options = None
    
    # Generowanie opcji tylko dla Plural MCQ (bo article_mcq usunięte)
    if item.get('assigned_task') == 'plural_mcq':
        correct = str(item.get('plural', ''))
        base = str(item.get('german_word', ''))
        
        # Algorytm generowania fałszywych odpowiedzi (dystraktorów)
        # Tworzy typowe niemieckie końcówki liczby mnogiej
        suffixes = ["n", "en", "e", "s", "er", "¨e", "¨er"]
        potential_distractors = []
        
        for s in suffixes:
            # Prosta symulacja: dodajemy końcówkę do słowa
            candidate = f"{base}{s}"
            # Jeśli wygenerowane słowo jest inne niż poprawne, dodajemy do listy
            if candidate.lower() != correct.lower() and candidate.lower() != base.lower():
                potential_distractors.append(candidate)
        
        # Jeśli z jakiegoś powodu lista pusta, dodaj cokolwiek
        if not potential_distractors:
            potential_distractors = [f"{base}en", f"{base}s"]

        # Wybierz 2 unikalne błędne odpowiedzi
        distractors = list(set(potential_distractors))
        if len(distractors) > 2:
            distractors = random.sample(distractors, 2)
            
        options = list(set([correct] + distractors))
        random.shuffle(options)

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
    
    if phase == "learning":
        session["current_index"] += 1
        return {"is_correct": True, "move_next": True}

    items = session["items"][phase]
    if session["current_index"] >= len(items):
        return {"is_correct": False, "move_next": False, "transition": True}

    item = items[session["current_index"]]
    
    # Walidacja odpowiedzi
    correct_val = ""
    if item.get('assigned_task') == 'type_word':
        # Wymagamy: Rodzajnik + Spacja + Słowo
        correct_val = f"{item.get('article','')} {item.get('german_word','')}"
    elif item.get('assigned_task') == 'plural_mcq':
        correct_val = item.get('plural','')
    
    # Porównanie (case-insensitive dla testu, można zmienić na sensitive usuwając .lower())
    is_correct = str(correct_val).strip().lower() == data.user_answer.strip().lower()
    
    if phase == "pre-test":
        session["logs"].append({
            "word": item.get('german_word'),
            "is_correct": is_correct,
            "user_input": data.user_answer,
            "task_type": item.get('assigned_task')
        })

    session["current_index"] += 1
    
    # Sprawdzenie końca fazy
    if phase == "pre-test" and session["current_index"] >= len(items):
        if session["condition"] == "B":
            path_b = await generate_adaptive_learning_path_B(session["items"]["learning_raw"], session["logs"])
            session["items"]["learning_path"] = path_b
        elif session["condition"] == "A" and not session["items"]["learning_path"]:
             session["items"]["learning_path"] = generate_static_learning_path_A(session["items"]["learning_raw"])
            
        session["phase"] = "learning"
        session["current_index"] = 0
        return {"is_correct": is_correct, "move_next": True, "transition": "learning", "feedback": f"Correct: {correct_val}"}

    return {"is_correct": is_correct, "move_next": True, "feedback": f"Correct: {correct_val}"}

@router.post("/experiment/skip")
async def skip_to_phase(data: SkipPhaseRequest):
    session = EXP_CACHE.get(data.session_id)
    if session:
        session["phase"] = data.phase
        session["current_index"] = 0
        if data.phase == "learning" and not session["items"]["learning_path"]:
             session["items"]["learning_path"] = generate_static_learning_path_A(session["items"]["learning_raw"])
    return {"status": "ok"}
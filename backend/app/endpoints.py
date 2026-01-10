import pandas as pd
import random
from uuid import uuid4
from typing import Dict, List, Set
from fastapi import APIRouter, HTTPException
from datetime import datetime

from .models import SessionInit, AnswerSubmit, SkipPhaseRequest, DemographicData, TutorRequest
from .llm_service import generate_static_learning_path_A, generate_adaptive_learning_path_B, generate_tutor_response
from .storage import init_db, save_session_result, create_session, get_session, update_session, delete_session

router = APIRouter()
try: init_db()
except: pass

try:
    VOCAB_DF = pd.read_csv("vocab.csv", sep=";")
    VOCAB_DF = VOCAB_DF.dropna(subset=['image_id', 'german_word'])
    VOCAB_DF = VOCAB_DF.where(pd.notnull(VOCAB_DF), "")
except: VOCAB_DF = pd.DataFrame()

# EXP_CACHE: Dict[str, dict] = {} # REMOVED: In-memory cache is not scalable

def get_valid_variations(article_str: str, word_str: str) -> Set[str]:
    valid = set()
    arts = [a.strip() for a in str(article_str).split('/')]
    words = [w.strip() for w in str(word_str).split('/')]
    for w in words:
        valid.add(w)
        for a in arts: valid.add(f"{a} {w}")
    if len(arts) == len(words) and len(words) > 1:
        for i in range(len(words)): valid.add(f"{arts[i]} {words[i]}")
    return valid

def prepare_experiment_items():
    scenarios = ["apartment_request", "travel", "swimming", "pet_sitting", "birthday"]
    pool = []
    for sc in scenarios:
        items = VOCAB_DF[VOCAB_DF['scenario'] == sc].to_dict('records')
        pool.extend(random.sample(items, 4) if len(items) >= 4 else items)
            
    test_set = random.sample(pool, 5)
    pre = [i.copy() for i in test_set]
    post = [i.copy() for i in test_set]
    
    types = ["type_word"] * 3 + ["plural_mcq"] * 2
    random.shuffle(types)
    for i in range(5):
        t = types[i]
        if t == "plural_mcq" and not str(pre[i].get('plural', '')).strip(): t = "type_word"
        pre[i]['assigned_task'] = t
        post[i]['assigned_task'] = t
    return test_set, pre, post

@router.post("/experiment/init")
async def init_experiment(data: SessionInit):
    sid = str(uuid4())
    _, pre, post = prepare_experiment_items()
    l_path = []
    # Save language preference if provided, otherwise default to "en" for now (frontend doesn't send it in Init yet)
    # We will assume "en" or check if we can get it later.
    # Actually, we should store it in session.
    
    if data.condition == "A":
        l_path = generate_static_learning_path_A(pre)

    sess_data = {
        "session_id": sid, 
        "user_id": data.user_id, 
        "condition": data.condition,
        "phase": "pre-test", 
        "current_index": 0,
        "items": { "pre-test": pre, "learning_raw": pre, "learning_path": l_path, "post-test": post },
        "logs": [],
        "start_time": datetime.now().isoformat() # Serialize for JSON
    }
    
    # Save to Supabase
    create_session(sid, sess_data)
    
    return {"session_id": sid}

@router.get("/experiment/trial/{session_id}")
async def get_trial(session_id: str):
    sess = get_session(session_id)
    if not sess: raise HTTPException(404, detail="Session not found")
    phase = sess["phase"]
    idx = sess["current_index"]

    # 1. Obsługa fazy Learning (bez zmian, to działało poprawnie)
    if phase == "learning":
        path = sess["items"]["learning_path"]
        if idx >= len(path):
            sess["phase"] = "post-test"
            sess["current_index"] = 0
            
            # Update DB before recursion to prevent infinite loop
            update_session(session_id, sess)
            
            return await get_trial(session_id)
        
        # Zabezpieczenie przed pustą ścieżką
        if not path:
             return {"status": "error", "message": "Learning path empty"}

        return {
            "phase": "learning", "index": idx, "total_in_phase": len(path),
            "task_type": "learning_step", "payload": path[idx]
        }

    # 2. Obsługa Pre-Test i Post-Test
    items = sess["items"][phase]
    
    # SPRAWDZENIE CZY KONIEC FAZY
    if idx >= len(items):
        # FIX: AUTOMATIC PHASE SWITCH
        # Jeśli skończył się pre-test, musimy PRZEŁĄCZYĆ na learning
        if phase == "pre-test":
            print(f">>> Switching session {session_id} from pre-test to learning")
            sess["phase"] = "learning"
            sess["current_index"] = 0
            
            # Update DB
            update_session(session_id, sess)
            
            # Rekurencyjne wywołanie, żeby od razu zwrócić pierwszy element nauki
            return await get_trial(session_id)
        
        elif phase == "post-test": 
            return {"status": "completed"}
        
        # Fallback (rzadki przypadek)
        return {"status": "transition"}

    item = items[idx]
    opts = None
    if item.get('assigned_task') == 'plural_mcq':
        corr = str(item.get('plural', '')).split('/')[0].strip()
        w1 = str(item.get('plural_wrong_1', '')).strip() or corr+"s"
        w2 = str(item.get('plural_wrong_2', '')).strip() or corr+"en"
        opts = list(set([corr, w1, w2]))
        while len(opts) < 3: opts.append(corr+"e")
        random.shuffle(opts)

    return {
        "phase": phase, "index": idx, "total_in_phase": len(items),
        "task_type": item.get('assigned_task', 'type_word'),
        "image_url": f"/images/{item['image_id']}.jpg",
        "english_gloss": item['english_gloss'],
        "options": opts
    }

@router.post("/experiment/submit")
async def submit_answer(data: AnswerSubmit):
    sess = get_session(data.session_id)
    if not sess: raise HTTPException(404, detail="Session not found")
    phase = sess["phase"]
    
    if phase == "learning":
        sess["current_index"] += 1
        update_session(data.session_id, sess)
        return {"is_correct": True, "move_next": True}

    items = sess["items"][phase]
    if sess["current_index"] >= len(items):
        return {"is_correct": False, "move_next": False, "transition": True}

    item = items[sess["current_index"]]
    user_ans = data.user_answer.strip()
    
    is_correct = False
    score = 0.0
    feedback_txt = ""
    
    # FIX VI: SCORING LOGIC
    if item.get('assigned_task') == 'plural_mcq':
        valid = [p.strip() for p in str(item.get('plural', '')).split('/')]
        if user_ans in valid:
            is_correct = True
            score = 1.0
        feedback_txt = f"Correct: {valid[0]}"
        
    elif item.get('assigned_task') == 'type_word':
        valid_set = get_valid_variations(item.get('article', ''), item.get('german_word', ''))
        
        # Exact match
        if user_ans in valid_set:
            is_correct = True
            score = 1.0
        # Case insensitive match (0.5 pts)
        else:
            valid_lower = {v.lower() for v in valid_set}
            if user_ans.lower() in valid_lower:
                is_correct = True
                score = 0.5
                feedback_txt = "Almost! Watch the case (Capital letters)."
            else:
                score = 0.0
        
        if not feedback_txt:
            feedback_txt = f"Correct: {str(item.get('article')).split('/')[0]} {str(item.get('german_word')).split('/')[0]}"

    log = {
        "phase": phase, "word": item.get('german_word'),
        "task_type": item.get('assigned_task'), "user_input": user_ans,
        "is_correct": is_correct, "score": score,
        "timestamp": str(datetime.now())
    }
    sess["logs"].append(log)
    sess["current_index"] += 1
    
    transition = False
    if sess["current_index"] >= len(items):
        transition = True
        if phase == "pre-test" and sess["condition"] == "B":
            print(">>> Triggering AI...")
            # We don't have language explicitly stored in session root reliably yet, but let's default to "en"
            # Or we can check if frontend sends it. Ideally, session info should have language.
            # For now, let's hardcode 'de' if context suggests, or better, change Init to accept it.
            # But the user complains about "DE, EN translation", so likely they want the UI language.
            # Let's assume 'en' is the default interface language, unless specified.
            # We will pass "de" as target_language causes the whole UI to be in German?
            # User said "tlumaczenie DE, EN nie dziala".
            # Let's pass 'en' as default, but we need to know what the user selected.
            # The tutor panel handles language switching. The Learning Path is generated ONCE.
            # Ideally we generate content in ENGLISH (as instructions) and German (as content).
            # Changing prompt to respect "target_language" helps.
            
            # Let's default to "en" for instructions unless we know otherwise.
            # If the user wants the interface in German, we need that info.
            # Currently frontend `startExperiment` doesn't send language.
            
            sess["items"]["learning_path"] = await generate_adaptive_learning_path_B(sess["items"]["learning_raw"], sess["logs"], target_language="en")
        elif phase == "pre-test" and sess["condition"] == "A" and not sess["items"]["learning_path"]:
             sess["items"]["learning_path"] = generate_static_learning_path_A(sess["items"]["learning_raw"])

    # Update DB
    update_session(data.session_id, sess)

    return {"is_correct": is_correct, "score": score, "move_next": True, "transition": transition, "feedback": feedback_txt}

@router.post("/experiment/finalize")
async def finalize_experiment(data: DemographicData):
    # 1. Pobieramy sesję z bazy
    sess = get_session(data.session_id)
    if not sess: 
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 2. Obliczamy czas trwania eksperymentu
    # Fallback na datetime.now() jeśli brak start_time
    try:
        start_time = datetime.fromisoformat(sess.get("start_time"))
    except:
        start_time = datetime.now()
        
    end_time = datetime.now()
    
    # Obliczamy różnicę w sekundach (rzutujemy na int, żeby nie mieć ułamków)
    duration_seconds = int((end_time - start_time).total_seconds())
    
    print(f"⏱️ User {sess.get('user_id')} finished in {duration_seconds} seconds.")

    # 3. Wysyłamy dane do zapisu w bazie (storage.py)
    # data.dict() zawiera to co przyszło z frontendu: age, gender, education, questionnaire itd.
    save_session_result(sess, data.dict(), duration_seconds)
    
    # 4. Czyścimy sesję z tabeli aktywnych sesji
    delete_session(data.session_id)
    
    return {"status": "success"}

@router.post("/experiment/skip")
async def skip_to_phase(data: SkipPhaseRequest):
    session = get_session(data.session_id)
    if session:
        session["phase"] = data.phase
        session["current_index"] = 0
        if data.phase == "learning" and not session["items"]["learning_path"]:
             session["items"]["learning_path"] = generate_static_learning_path_A(session["items"]["learning_raw"])
        update_session(data.session_id, session)
    return {"status": "ok"}

@router.post("/experiment/tutor/ask")
async def ask_tutor(data: TutorRequest):
    # Context is passed directly from frontend
    response = await generate_tutor_response(
        question=data.question,
        context=data.task_context.dict(),
        is_nudge=False,
        target_language=data.response_language
    )
    
    # LOGGING
    try:
        sess = get_session(data.session_id)
        if sess:
            tutor_log = {
                "timestamp": datetime.now().isoformat(),
                "user_question": data.question,
                "context": data.task_context.dict(),
                "tutor_response": response
            }
            # Initialize list if not exists
            if "tutor_logs" not in sess: sess["tutor_logs"] = []
            sess["tutor_logs"].append(tutor_log)
            
            update_session(data.session_id, sess)
    except Exception as e:
        print(f"⚠️ Failed to log tutor interaction: {e}")
        
    return response
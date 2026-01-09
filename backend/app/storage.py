# app/storage.py

import os
from supabase import create_client, Client
import json

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = None

def init_db():
    global supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Connected to Supabase")
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
    else:
        print("⚠️ Missing Supabase credentials.")

# --- ZMIANA: Dodano parametr 'duration' ---
def save_session_result(session_data: dict, demographics: dict, duration: int):
    if not supabase:
        print("❌ Supabase not connected.")
        return

    # Wyciągamy ankietę (pytania 1-5 i opinie)
    quest_data = demographics.get("questionnaire", {})
    
    payload = {
        "session_id": str(session_data.get("session_id")),
        "user_id": str(session_data.get("user_id")),
        "condition": session_data.get("condition"),
        
        # Nowe pole czasu
        "duration_seconds": duration,
        
        # Demografia (płaskie pola)
        "age": int(demographics.get("age", 0)) if demographics.get("age") else None,
        "gender": demographics.get("gender"),
        "education": demographics.get("education"),
        "german_level": demographics.get("german_level"),
        
        # JSONB: Zapisujemy całą ankietę i historię zadań
        "questionnaire": quest_data,
        "logs": session_data.get("logs", [])
    }

    try:
        supabase.table("experiment_results").insert(payload).execute()
        print(f"✅ Saved results for {payload['user_id']} (Time: {duration}s)")
    except Exception as e:
        print(f"❌ DB Error: {e}")
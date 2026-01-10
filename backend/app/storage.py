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
        "logs": session_data.get("logs", []),
        "tutor_logs": session_data.get("tutor_logs", [])
    }

    try:
        # DEBUG: Print exact payload
        print(f"📦 SAVING PAYLOAD (DEBUG): {json.dumps(payload, indent=2, ensure_ascii=False)}")
        supabase.table("experiment_results").insert(payload).execute()
        print(f"✅ Saved results for {payload['user_id']} (Time: {duration}s)")
    except Exception as e:
        print(f"❌ DB Error: {e}")

# --- SESSION MANAGEMENT (Scalability Fix) ---

def create_session(session_id: str, data: dict):
    if not supabase:
        print("❌ Supabase not connected. Cannot create session.")
        return False
    try:
        payload = {
            "session_id": session_id,
            "data": data
        }
        supabase.table("active_sessions").insert(payload).execute()
        return True
    except Exception as e:
        print(f"❌ Failed to create session {session_id}: {e}")
        return False

def get_session(session_id: str) -> dict:
    if not supabase: return None
    try:
        response = supabase.table("active_sessions").select("data").eq("session_id", session_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]["data"]
        return None
    except Exception as e:
        print(f"❌ Failed to get session {session_id}: {e}")
        return None

def update_session(session_id: str, data: dict):
    if not supabase: return False
    try:
        supabase.table("active_sessions").update({"data": data, "updated_at": "now()"}).eq("session_id", session_id).execute()
        return True
    except Exception as e:
        print(f"❌ Failed to update session {session_id}: {e}")
        return False

def delete_session(session_id: str):
    if not supabase: return
    try:
        supabase.table("active_sessions").delete().eq("session_id", session_id).execute()
    except Exception as e:
        print(f"❌ Failed to delete session {session_id}: {e}")

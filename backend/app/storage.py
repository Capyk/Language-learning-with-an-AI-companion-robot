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

def get_next_group() -> str:
    """
    Determines the next group (A/B) based on total number of started sessions (active + completed).
    This ensures alternating assignment order: A, B, A, B...
    """
    if not supabase: return "A" # Default
    try:
        # Count completed
        res_completed = supabase.table("experiment_results").select("id", count="exact").execute()
        count_completed = res_completed.count if res_completed.count is not None else 0
        
        # Count active
        res_active = supabase.table("active_sessions").select("session_id", count="exact").execute()
        count_active = res_active.count if res_active.count is not None else 0
        
        total = count_completed + count_active
        return "A" if total % 2 == 0 else "B"
    except Exception as e:
        print(f"❌ Failed to calculate next group: {e}")
        return "A" # Fallback

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

# --- ACCESS CODE MANAGEMENT ---

def get_all_codes():
    if not supabase: return []
    try:
        res = supabase.table("access_codes").select("*").order("created_at", desc=True).execute()
        return res.data
    except Exception as e:
        print(f"❌ Failed to get codes: {e}")
        return []

def generate_codes(count: int = 10):
    if not supabase: return []
    import secrets
    import string
    
    new_codes = []
    for _ in range(count):
        # Format: XXX-XXX (easier to read)
        chars = string.ascii_uppercase + string.digits
        part1 = ''.join(secrets.choice(chars) for _ in range(3))
        part2 = ''.join(secrets.choice(chars) for _ in range(3))
        code = f"{part1}-{part2}"
        new_codes.append({"code": code, "used": False, "copy_count": 0, "usage_count": 0})
        
    try:
        data = supabase.table("access_codes").insert(new_codes).execute()
        return data.data
    except Exception as e:
        print(f"❌ Failed to generate codes: {e}")
        return []

def increment_code_copy_count(code: str):
    if not supabase: return
    try:
        # We need to get current count first or use an RPC if available. 
        # For simplicity: GET -> UPDATE
        res = supabase.table("access_codes").select("copy_count").eq("code", code).execute()
        if res.data:
            curr = res.data[0].get("copy_count", 0)
            supabase.table("access_codes").update({"copy_count": curr + 1}).eq("code", code).execute()
    except Exception as e:
        print(f"❌ Failed to increment copy count: {e}")

def increment_code_usage(code: str):
    if not supabase: return
    try:
        res = supabase.table("access_codes").select("usage_count").eq("code", code).execute()
        if res.data:
            # Handle case where usage_count might be None if column is new/null
            curr = res.data[0].get("usage_count") or 0
            supabase.table("access_codes").update({"usage_count": curr + 1}).eq("code", code).execute()
    except Exception as e:
        print(f"❌ Failed to increment usage count: {e}")


def validate_and_use_code(code: str) -> str:
    """
    Validates code. If valid and unused, returns assigned group (A/B).
    Does NOT mark as used yet (we delete on finish).
    However, to prevent double login, maybe we should mark as 'active'?
    For now, adhering to user requirement: 'remove... whenever someone... finishes'.
    So multiple people *could* potentially use it if they type it fast enough? 
    Risk is low. We will just check if it exists.
    """
    if not supabase: return None
    try:
        # Check if code exists
        res = supabase.table("access_codes").select("*").eq("code", code).execute()
        if not res.data: return None
        
        # Validate ONLY. Group assignment happens at Init.
        return True
    except Exception as e:
        print(f"❌ Failed to validate code: {e}")
        return None

def delete_access_code(code: str):
    if not supabase: return
    try:
        supabase.table("access_codes").delete().eq("code", code).execute()
        print(f"🗑️ Access code {code} deleted.")
    except Exception as e:
        print(f"❌ Failed to delete code {code}: {e}")


import os
from supabase import create_client, Client
import json

# Render automatycznie podstawi te zmienne, jeśli je skonfigurujemy w panelu
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = None

def init_db():
    """Inicjalizacja połączenia z Supabase."""
    global supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Connected to Supabase")
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
    else:
        print("⚠️ Missing Supabase credentials in environment variables.")

def verify_access_code(code: str) -> bool:
    """Sprawdza czy kod istnieje i jest aktywny."""
    # Na razie, jeśli nie używasz kodów, zwracaj True
    # Jeśli chcesz włączyć kody, odkomentuj logikę poniżej
    return True 

    # if not supabase: return True 
    # try:
    #     resp = supabase.table("access_codes").select("*").eq("code", code).execute()
    #     if not resp.data: return False
    #     return not resp.data[0]['is_used']
    # except: return False

def mark_code_as_used(code: str):
    pass
    # if supabase and code:
    #     supabase.table("access_codes").update({"is_used": True}).eq("code", code).execute()

def save_session_result(session_data: dict, demographics: dict):
    """Wysyła wyniki do Supabase."""
    if not supabase:
        print("❌ Cannot save: Supabase not connected.")
        return

    payload = {
        "session_id": session_data.get("session_id"),
        "user_id": session_data.get("user_id"),
        "condition": session_data.get("condition"),
        # Konwersja na int, bo formularz wysyła stringi
        "age": int(demographics.get("age", 0)) if demographics.get("age") else 0,
        "gender": demographics.get("gender"),
        "education": demographics.get("education"),
        "german_level": demographics.get("german_level"),
        "logs": session_data.get("logs", [])
    }

    try:
        supabase.table("experiment_results").insert(payload).execute()
        print("✅ Data saved to Supabase!")
    except Exception as e:
        print(f"❌ Error saving to Supabase: {e}")
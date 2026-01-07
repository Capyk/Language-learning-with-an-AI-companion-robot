import sqlite3
import json
import os

DB_NAME = "experiment_data.db"

# def init_db():
#     """Tworzy tabelę w bazie danych, jeśli nie istnieje."""
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
    
#     # Tabela wyników
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS results (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             session_id TEXT,
#             user_id TEXT,
#             condition TEXT,
#             age INTEGER,
#             gender TEXT,
#             education TEXT,
#             german_level TEXT,
#             start_time TEXT,
#             end_time TEXT,
#             logs_json TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

# def save_session_result(session_data: dict, demographics: dict):
#     """Zapisuje zakończoną sesję do bazy."""
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
    
#     logs_json = json.dumps(session_data.get("logs", []), ensure_ascii=False)
    
#     c.execute('''
#         INSERT INTO results (
#             session_id, user_id, condition, 
#             age, gender, education, german_level,
#             start_time, end_time, logs_json
#         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#     ''', (
#         session_data.get("session_id"),
#         session_data.get("user_id"),
#         session_data.get("condition"),
#         demographics.get("age"),
#         demographics.get("gender"),
#         demographics.get("education"),
#         demographics.get("german_level"),
#         str(datetime.now()), # End time (approx)
#         str(datetime.now()), # Placeholder, można dodać start_time do sesji
#         logs_json
#     ))
    
#     conn.commit()
#     conn.close()
# app/storage.py
# Wersja "Offline" / Mock - nie łączy się z bazą, zawsze zwraca sukces.

# app/storage.py
# Wersja do testów lokalnych - tylko wypisuje dane w konsoli.

def init_db():
    print("⚠️ DB disabled for local testing (Mock Mode)")

def verify_access_code(code: str) -> bool:
    return True

def mark_code_as_used(code: str):
    pass

def save_session_result(session_data: dict, demographics: dict):
    print("\n========= SAVING SESSION DATA (MOCK) =========")
    print(f"User ID: {session_data.get('user_id')}")
    print(f"Condition: {session_data.get('condition')}")
    print(f"Demographics: {demographics}")
    print(f"Logs Count: {len(session_data.get('logs', []))}")
    print("==============================================\n")
# Import datetime here to avoid circular imports elsewhere if needed
# from datetime import datetime
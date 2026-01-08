from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware 
from google import genai

from . import endpoints
import app.llm_service as llm_service

load_dotenv()

app = FastAPI(title="AI Tutor Backend - Experiment Version")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Zezwól wszystkim (dla pewności)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES CONFIGURATION & DEBUG ---
# Zakładamy, że folder 'data' jest w głównym katalogu projektu (obok requirements.txt)
IMAGE_DIRECTORY = "data/images"

# --- DIAGNOSTYKA PLIKÓW (To nam powie prawdę w logach) ---
print(f"📂 Sprawdzam folder: {os.path.abspath(IMAGE_DIRECTORY)}")
if os.path.exists(IMAGE_DIRECTORY):
    files = os.listdir(IMAGE_DIRECTORY)
    print(f"✅ Folder istnieje! Znaleziono {len(files)} plików.")
    if len(files) > 0:
        print(f"👀 Przykładowe pliki: {files[:5]}") # Wypisz pierwsze 5 plików
    else:
        print("⚠️ Folder jest PUSTY! (Sprawdź .gitignore)")
else:
    print(f"❌ BŁĄD KRYTYCZNY: Folder '{IMAGE_DIRECTORY}' NIE ISTNIEJE na serwerze!")
    # Spróbujmy znaleźć gdzie on jest
    print("🔍 Listuję obecny katalog roboczy:")
    print(os.listdir("."))
    if os.path.exists("data"):
        print("🔍 Listuję folder 'data':")
        print(os.listdir("data"))
# ---------------------------------------------------------

# Montowanie folderu (Musi być dokładnie tak)
# URL: /images/... -> Folder na dysku: data/images/...
app.mount("/images", StaticFiles(directory=IMAGE_DIRECTORY), name="images")

app.include_router(endpoints.router)

@app.on_event("startup")
async def startup_event():
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("🛑 Error: GEMINI_API_KEY not found.")
            llm_service.CLIENT_INITIALIZED = False
        else:
            llm_service.CLIENT_INITIALIZED = True
            print("✅ Gemini Client Configuration Loaded.")
    except Exception as e:
        print(f"🛑 Critical error initializing Gemini Client: {e}")
        llm_service.CLIENT_INITIALIZED = False

    print(f"🚀 Server is running.")
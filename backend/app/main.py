# /backend/app/main.py

from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# --- NOWY IMPORT ---
from fastapi.middleware.cors import CORSMiddleware 
from google import genai

# Internal imports
from . import endpoints
import app.llm_service as llm_service

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Tutor Backend - Experiment Version")

# --- CORS CONFIGURATION (KLUCZOWA ZMIANA) ---
# Pozwalamy na połączenia z dowolnego miejsca (dla testów) lub konkretnie z Vercel
origins = [
    "http://localhost:5173",  # Twój lokalny frontend
    "https://german-learning-language-backend.onrender.com", # Twój backend
    "*" # Dopuszcza wszystko (najbezpieczniej na start, żeby zadziałało)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Zmienione na "*" aby na pewno zadziałało z Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------

# --- Static Files Configuration ---
IMAGE_DIRECTORY = "data/images"

if not os.path.exists(IMAGE_DIRECTORY):
    print(f"⚠️ Warning: The directory '{IMAGE_DIRECTORY}' does not exist. Image serving may fail.")
else:
    # Upewniamy się, że katalog istnieje, żeby uniknąć błędu 500
    app.mount("/images", StaticFiles(directory=IMAGE_DIRECTORY), name="images")

# --- Router Setup ---
app.include_router(endpoints.router)

# --- Application Startup Hook ---
@app.on_event("startup")
async def startup_event():
    """
    Executes setup logic when the FastAPI server starts.
    """
    
    # 1. Initialize the Gemini Client for the LLM service
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("🛑 Error: GEMINI_API_KEY not found in environment variables.")
            llm_service.CLIENT_INITIALIZED = False
        else:
            # Przekazujemy klucz, ale inicjalizacja właściwa nastąpi w llm_service
            # jeśli używamy google-genai, obiekt klienta tworzymy tam
            llm_service.CLIENT_INITIALIZED = True
            print("✅ Gemini Client Configuration Loaded.")
    except Exception as e:
        print(f"🛑 Critical error initializing Gemini Client: {e}")
        llm_service.CLIENT_INITIALIZED = False

    print(f"🚀 Server is running.")
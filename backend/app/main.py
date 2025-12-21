# /backend/app/main.py

from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from google import genai

# Internal imports
from . import endpoints
import app.llm_service as llm_service

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Tutor Backend - Experiment Version")

# --- Static Files Configuration ---
# This allows the frontend to access images via URLs like http://localhost:8000/images/img_01.png
# We map the physical directory 'data/images' to the URL path '/images'
IMAGE_DIRECTORY = "data/images"

if not os.path.exists(IMAGE_DIRECTORY):
    print(f"⚠️ Warning: The directory '{IMAGE_DIRECTORY}' does not exist. Image serving may fail.")

app.mount("/images", StaticFiles(directory=IMAGE_DIRECTORY), name="images")

# --- Router Setup ---
# Include the experiment and labeling endpoints
app.include_router(endpoints.router)

# --- Application Startup Hook ---

@app.on_event("startup")
async def startup_event():
    """
    Executes setup logic when the FastAPI server starts.
    This includes initializing the Gemini client and checking environment health.
    """
    
    # 1. Initialize the Gemini Client for the LLM service
    try:
        # Assign the global client object in llm_service module
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("🛑 Error: GEMINI_API_KEY not found in environment variables.")
            llm_service.CLIENT_INITIALIZED = False
        else:
            llm_service.client = genai.Client(api_key=api_key)
            llm_service.CLIENT_INITIALIZED = True
            print("✅ Gemini Client Initialized Successfully.")
    except Exception as e:
        print(f"🛑 Critical error initializing Gemini Client: {e}")
        llm_service.CLIENT_INITIALIZED = False

    # 2. Log server status
    print(f"🚀 Server is running. Static images served from: {os.path.abspath(IMAGE_DIRECTORY)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "message": "AI Tutor Backend is running."}
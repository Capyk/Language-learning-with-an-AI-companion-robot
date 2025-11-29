# /backend/app/main.py

from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
import os
import redis.asyncio as redis 
from google import genai
from . import endpoints
from .llm_service import client, CLIENT_INITIALIZED # Import the placeholder variables
import app.llm_service as llm_service # Import the module to call the function

# --- Redis Initialization (Unchanged) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
# redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True) 

app = FastAPI(title="AI Tutor Backend")
app.include_router(endpoints.router)

# --- Application Startup Hook ---

@app.on_event("startup")
async def startup_event():
    """Executes necessary setup logic after the app starts loading."""
    # 1. Initialize the Gemini Client (CRITICAL STEP)
    try:
        # Assign the global client object in llm_service
        llm_service.client = genai.Client()
        llm_service.CLIENT_INITIALIZED = True
        print("✅ Gemini Client Initialized Successfully.")
    except Exception as e:
        print(f"🛑 Error initializing Gemini Client: {e}")
        llm_service.CLIENT_INITIALIZED = False
        
    # 2. Verify Redis connection
    # try:
    #     await redis_client.ping()
    #     print(f"✅ Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    # except Exception as e:
    #     print(f"🛑 Could not connect to Redis: {e}")

# Inject Redis client
# app.state.redis = redis_client

@app.get("/")
def read_root():
    return {"message": "AI Tutor Backend is Live"}
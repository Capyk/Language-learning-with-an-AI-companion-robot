from fastapi import FastAPI
from . import endpoints
# /backend/app/main.py
from dotenv import load_dotenv
import os

# Load environment variables fro    m .env file into the system environment
load_dotenv() 

# Check if the key is loaded (for debugging)
# print(os.getenv("GEMINI_API_KEY"))

# Initialize the FastAPI application
app = FastAPI(title="Pepper AI Tutor Backend")
# ... rest of your code

# Initialize the FastAPI application
app = FastAPI(title="Pepper AI Tutor Backend")

# Include the endpoints defined in endpoints.py
app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Pepper Tutor Backend is Running"}
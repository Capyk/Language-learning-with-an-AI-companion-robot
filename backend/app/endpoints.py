# /backend/app/endpoints.py

from fastapi import APIRouter, HTTPException
from typing import List, Dict
import random

# Mock data for images and labels
MOCK_IMAGES = [
    {"url": "http://img.com/dog.jpg", "german_label": "der Hund"},
    {"url": "http://img.com/table.jpg", "german_label": "der Tisch"},
    {"url": "http://img.com/book.jpg", "german_label": "das Buch"},
]

router = APIRouter()

# --- Endpoint 1: Get Initial Task (Presentation Phase) ---
@router.get("/task/get_presentation", response_model=Dict)
async def get_initial_presentation():
    """Provides a list of 5 images and their correct German labels."""
    
    # Select 5 random images for the quiz
    selected_tasks = random.sample(MOCK_IMAGES, 5)
    
    # In a real system, you would save these 5 correct labels to Redis here,
    # along with a unique session ID, before returning the data.
    
    return {
        "session_id": "ABC12345", # Placeholder ID
        "tasks": selected_tasks
    }
# /backend/app/endpoints.py (Final Version without Redis)

from fastapi import APIRouter, HTTPException, Request # Request remains for future use
from typing import Dict, List
from uuid import uuid4
import random
import json
# Removed: import redis.asyncio as redis 

# Import the necessary schemas and LLM/Utility functions
from .models import TaskType, TaskRequest, TaskResponse, ImageVocabItem 
from .llm_service import generate_vocab_task, validate_answer, generate_gec_task, real_time_correction 

router = APIRouter()

# --- TEMPORARY IN-MEMORY CACHE (REPLACING REDIS) ---
# NOTE: Data will be lost when the server restarts. Use only for demo/testing.
TEMP_SESSION_CACHE: Dict[str, dict] = {} 
# ---

# --- MOCK DATA SOURCE (Image Labeling) ---
MOCK_IMAGE_DATA = [
    {"url": "https://st5.depositphotos.com/1007566/67451/v/450/depositphotos_674511126-stock-illustration-meat-product-sausage-icon-isolated.jpg", "german": "die Wurst", "english": "sausage", "topic": "food"},
    {"url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT_P7IXme1ALavzmagyBiW8JWtOUg6-GgkaZw&s", "german": "der Käse", "english": "cheese", "topic": "food"},
    {"url": "https://previews.123rf.com/images/tassiatk/tassiatk2206/tassiatk220600066/187484027-old-car-drawing-isolated-vector-retro-red-auto-without-roof-cabriolet-nastolgia-illustration.jpg", "german": "das Auto", "english": "car", "topic": "transport"},
    {"url": "https://www.shutterstock.com/shutterstock/photos/2485852297/display_1500/stock-vector-standing-lion-african-wild-animal-lion-walking-profile-body-side-view-vector-2485852297.jpg", "german": "der Löwe", "english": "lion", "topic": "animals"},
    {"url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSw3MMFWMwUYtn64beey1ExhARN2U7iV2Rflg&s", "german": "das Gras", "english": "grass", "topic": "garden"},
]

# --- Internal Generator Function ---
def _generate_image_labeling_task(num_items: int) -> List[ImageVocabItem]:
    """Selects and formats the image and vocabulary pairs."""
    selected_items = random.sample(MOCK_IMAGE_DATA, min(num_items, len(MOCK_IMAGE_DATA)))
    
    task_list = []
    for item in selected_items:
        task_list.append(
            ImageVocabItem(
                image_url=item["url"],
                german_label=item["german"],
                english_translation=item["english"]
            )
        )
    return task_list


@router.post("/task/create", response_model=TaskResponse)
async def create_new_task(request: TaskRequest):
    """Creates a new learning task based on the requested type and parameters."""
    
    # -----------------------------------------------------
    # Logic for IMAGE_LABELING
    # -----------------------------------------------------
    if request.task_type == TaskType.IMAGE_LABELING:
        task_payload_items = _generate_image_labeling_task(request.num_items)
        session_id = str(uuid4())
        
        # Save the state to the temporary in-memory dictionary
        TEMP_SESSION_CACHE[session_id] = {
            "user_id": request.user_id,
            "task_type": request.task_type.value,
            # Convert Pydantic models to dicts for simple JSON storage
            "items": [item.dict() for item in task_payload_items], 
            "current_item_index": 0 
        }
        
        return TaskResponse(
            session_id=session_id,
            task_type=request.task_type,
            payload={"image_items": task_payload_items} 
        )
    
    # -----------------------------------------------------
    # Logic for VOCABULARY_GENERATION
    # -----------------------------------------------------
    elif request.task_type == TaskType.VOCABULARY_GENERATION:
        
        # 1. Generate the vocabulary using the LLM Service (This is the slow, essential call)
        vocab_items = await generate_vocab_task(
            request.user_id, 
            request.num_items, 
            request.topic
        )
        
        session_id = str(uuid4())
        
        # Save the state to the temporary in-memory dictionary
        session_data = {
            "user_id": request.user_id, 
            # Convert Pydantic models to dicts for simple JSON storage
            "items": [item.dict() for item in vocab_items], 
            "current_item_index": 0
        }
        TEMP_SESSION_CACHE[session_id] = session_data
        
        # NOTE: The return type must match the payload structure from your models.py
        return TaskResponse(
            session_id=session_id, 
            task_type=request.task_type, 
            payload={"vocabulary_list": vocab_items}
        )

    # -----------------------------------------------------
    # Logic for GEC_CHALLENGE
    # -----------------------------------------------------
    elif request.task_type == TaskType.GEC_CHALLENGE:
        challenge_data = await generate_gec_task(request.user_id, request.difficulty_level, request.topic)
        session_id = str(uuid4())
        
        # Save the challenge and expected answer to the temporary cache
        TEMP_SESSION_CACHE[session_id] = {
            "user_id": request.user_id, 
            "expected_answer": challenge_data["expected_correction"], # Save the ground truth
            "flawed_sentence": challenge_data["flawed_sentence"]
        }
        
        return TaskResponse(
            session_id=session_id,
            task_type=request.task_type,
            payload={"challenge_sentence": challenge_data["flawed_sentence"]}
        )
        
    else:
        raise HTTPException(status_code=400, detail=f"Task type {request.task_type.value} not implemented or unknown.")


@router.post("/correction/live")
async def process_live_correction(
    user_id: str, 
    user_input: str, 
    difficulty: str = "A2"
):
    """Receives free-form user text and returns GEC correction and feedback."""
    
    correction_result = await real_time_correction(
        user_input, user_id, difficulty
    )
    
    return {
        "user_input": user_input,
        "correction": correction_result["corrected_text"],
        "tip": correction_result["tip"]
    }


@router.get("/image-labeling/{index}")
async def get_image_for_labeling(index: int):
    """
    Fetches image data and both German/English labels from MOCK_IMAGE_DATA 
    based on the index.
    """
    if index < 0 or index >= len(MOCK_IMAGE_DATA):
        raise HTTPException(status_code=404, detail="Image index out of bounds.")
    
    image_data = MOCK_IMAGE_DATA[index]
    
    return {
        "image_url": image_data["url"],
        "topic": image_data["topic"],
        "german_label": image_data["german"],
        "english_label": image_data["english"]
    }


@router.post("/image-labeling/analyze")
async def analyze_user_label(data: dict):
    """
    Accepts the user's label and input, sends the user_input to Gemini 
    via real_time_correction for detailed tip generation, and returns the feedback.
    """
    user_label = data.get("user_label")
    user_input = data.get("user_input")
    
    if not user_label or not user_input:
        raise HTTPException(status_code=400, detail="Both 'user_label' and 'user_input' are required.")
    
    # 1. Check for correctness locally (before calling Gemini)
    # This ensures the final "correct_label" is the ground truth from MOCK_IMAGE_DATA
    is_correct = user_label.strip().lower() == user_input.strip().lower()
    
    # 2. Call Gemini for the tip. We send the user's input AND the correct label
    # to give the LLM enough context to provide a specific tip (e.g., "wrong gender").
    # NOTE: Your real_time_correction function in llm_service.py must be updated
    # to accept and use the 'expected_answer' (user_label) in its prompt.
    gemini_response = await real_time_correction(
        user_input, 
        user_id="user_id_placeholder", 
        difficulty="A2",
        expected_answer=user_label # <-- PASS THE CORRECT ANSWER FOR CONTEXT
    )
    
    # 3. Assemble the final feedback structure
    if is_correct:
        correction_text = "Correct!"
        tip_text = "Excellent! You got the noun and the article right."
    else:
        # Use the feedback generated by Gemini for the correction message and detailed tip
        correction_text = gemini_response["corrected_text"]
        tip_text = gemini_response["tip"]
        
    feedback = {
        "correct_label": user_label,  # Use the ground truth correct label
        "user_input": user_input,
        "correction": correction_text,
        "tip": tip_text # <-- THIS IS THE TIP GENERATED BY GEMINI
    }
    
    return feedback
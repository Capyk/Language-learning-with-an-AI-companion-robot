# /backend/app/llm_service.py (Updated)

import asyncio
from typing import List, Dict
import json
import os
import httpx # Used for any external synchronous calls if needed, though GenAI SDK is better

# --- Gemeni API Setup ---
# ACTION: You must set your API Key in your environment variables!
# The SDK will automatically detect it.
# export GEMINI_API_KEY='YOUR_API_KEY'
from google import genai
from google.genai import types

# --- TEMPORARY IN-MEMORY STATE (SIMULATING REDIS) ---
TEMP_TASK_CACHE: Dict[str, dict] = {} 

# Initialize the Gemini Client outside of the async function for efficiency
# It will use the API key from your environment variable
try:
    client = genai.Client()
except Exception:
    # Fallback to ensure the script runs even if the key is missing initially
    print("Warning: Gemini client failed to initialize. Check GEMINI_API_KEY environment variable.")
    client = None 

def build_system_prompt(num_items: int):
    """Builds the system prompt for structured JSON output."""
    return f"""You are an assistant selecting exactly {num_items} thematically related German vocabulary words 
    and their English translations, suitable for an A2 level.
    You MUST output your answer exclusively as a valid JSON array.
    Your answer must strictly adhere to the following JSON format. Do NOT include any explanations, markdown, or text outside the JSON structure:
    [
      {{"german": "der Apfel", "english": "apple"}},
      {{"german": "der Baum", "english": "tree"}},
      ...
    ]"""

def build_user_prompt(topic: str):
    """Builds the user prompt to guide the vocabulary selection."""
    return f"Wähle {topic}-bezogene Wörter."


async def generate_vocab_task(user_id: str, num_items: int, topic: str) -> List[Dict]:
    """
    Calls the Gemini API to get structured vocabulary, handling retries and parsing.
    """
    if client is None:
        raise ConnectionError("Gemini API Client is not initialized. Check API Key.")
        
    system_prompt = build_system_prompt(num_items)
    user_prompt = build_user_prompt(topic)
    
    try:
        # Use a model capable of strong reasoning and JSON output (e.g., gemini-2.5-pro)
        response = await client.models.generate_content_async(
            model='gemini-2.5-pro', 
            contents=[
                types.Content(role="system", parts=[types.Part.from_text(system_prompt)]),
                types.Content(role="user", parts=[types.Part.from_text(user_prompt)])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "german": types.Schema(type=types.Type.STRING),
                            "english": types.Schema(type=types.Type.STRING),
                        },
                        required=["german", "english"]
                    )
                )
            )
        )
        
        # The API is requested to return JSON directly in response.text
        json_output = response.text.strip()
        
        # Parse the JSON response
        vocabulary = json.loads(json_output)
        
        # Final validation check
        if not isinstance(vocabulary, list) or len(vocabulary) != num_items:
            raise ValueError(f"LLM returned invalid number of items: expected {num_items}")

        return vocabulary
        
    except ConnectionError:
        raise ConnectionError("Gemini API ist nicht erreichbar.")
    except json.JSONDecodeError:
        # Handle cases where the model might fail the strict JSON constraint
        print(f"JSON Decode Error. Raw LLM response: {response.text}")
        raise ValueError("LLM returned non-parsable JSON. Review prompt.")
    except Exception as e:
        # Catch other API/runtime errors
        raise Exception(f"Unexpected Gemini API Error: {str(e)}")

# Keep the simple validation logic for the quiz phase separate
async def validate_answer(user_answer: str, correct_answer: str, user_id: str) -> Dict:
    # ... (Your existing validation logic goes here, no LLM required for simple quiz check)
    pass
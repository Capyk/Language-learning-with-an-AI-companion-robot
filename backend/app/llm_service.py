import asyncio
from typing import List, Dict
import json
from google import genai
from google.genai import types

# --- Client Placeholder (Set by main.py startup hook) ---
client = None 
CLIENT_INITIALIZED = False
# ---

# --- Utility Functions ---

def build_vocab_system_instruction(num_items: int):
    """Builds the system instruction for structured JSON vocabulary output."""
    return f"""You are an assistant selecting exactly {num_items} thematically related German vocabulary words 
    and their English translations, suitable for an A2 level. You MUST output your answer exclusively as a valid JSON array.
    Your answer must strictly adhere to the defined JSON schema. Do NOT include any explanations or text outside the JSON structure."""

def build_gec_system_instruction(difficulty: str, task_type: str):
    """Builds the system instruction for GEC generation/correction tasks."""
    if task_type == 'GEC_CHALLENGE':
        # Prompt for generating a flawed sentence (Challenge)
        return (f"You are a generator of German language challenges. Create ONE complex, full German sentence "
                f"at the {difficulty} level that contains a common grammatical error (e.g., incorrect case, "
                f"article, or verb placement). Return ONLY the flawed sentence and the corresponding correction.")
    
    elif task_type == 'VOCAB_CORRECTION': # <-- NEW TASK TYPE FOR VOCAB MISTAKES
        return (f"You are a specialized German vocabulary correction assistant for learners at the {difficulty} level. "
                f"Your task is to compare the user's input to the correct German vocabulary word and provide feedback. "
                f"Analyze the mistake (e.g., wrong article, wrong spelling). "
                f"Provide a concise 'corrected_text' (e.g., 'Incorrect article') and a detailed, helpful **tip in English** "
                f"in the 'tip' field explaining the specific mistake (e.g., 'Remember 'Käse' is masculine, so use 'der').")

    else: # TaskType is 'CORRECTION' (real_time_correction for GEC)
        # Prompt for correcting user input
        return (f"You are a high-precision German GEC (Grammatical Error Correction) engine. "
                f"The user is a learner at the {difficulty} level. Your task is to correct the user's text. "
                f"Provide ONLY the completely corrected sentence as the 'corrected_text' and a brief, encouraging feedback tip in the 'tip' field."
                f"Then, provide a brief, encouraging feedback **tip in English** in the 'tip' field.")

def build_user_prompt(topic: str):
    """Builds the simple user prompt to guide the topic selection."""
    return f"Wähle {topic}-bezogene Wörter."


async def generate_vocab_task(user_id: str, num_items: int, topic: str) -> List[Dict]:
    """Calls the Gemini API to get structured vocabulary."""
    
    if not CLIENT_INITIALIZED:
        raise ConnectionError("Gemini API Client failed to initialize. Check GEMINI_API_KEY.")
        
    user_prompt = build_user_prompt(topic)
    loop = asyncio.get_event_loop()
    
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model='gemini-2.5-pro', 
                contents=[types.Content(role="user", parts=[types.Part(text=user_prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=build_vocab_system_instruction(num_items), 
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={"german": types.Schema(type=types.Type.STRING), "english": types.Schema(type=types.Type.STRING)},
                            required=["german", "english"]
                        )
                    )
                )
            )
        )
        
        vocabulary = json.loads(response.text.strip())
        
        if not isinstance(vocabulary, list) or len(vocabulary) != num_items:
            raise ValueError(f"LLM returned invalid number of items: expected {num_items}")

        return vocabulary
        
    except Exception as e:
        print(f"Gemini API Execution Error: {e}")
        raise Exception(f"LLM Task Failure: {str(e)}")


async def real_time_correction(user_input: str, expected_answer: str, attempt_number: int, difficulty: str) -> dict:
    """
    Generates adaptive hints for Condition B based on the attempt number.
    This replaces the general purpose correction with a specific pedagogical strategy.
    """
    
    if not CLIENT_INITIALIZED:
        return {"tip": "Error: AI not initialized."}

    # Strategy: Change the depth of the hint based on how many times the user failed
    if attempt_number == 1:
        # Subtle hint: Focus on the type of error
        hint_depth = (
            f"The user is trying to learn the word '{expected_answer}'. "
            "Give a very subtle hint in English about the error (spelling, article, or plural). "
            "Do NOT mention the correct word at all."
        )
    elif attempt_number == 2:
        # Strong hint: Give structural help
        hint_depth = (
            f"The user failed twice to get '{expected_answer}'. "
            f"Give a strong hint. You can mention that it starts with '{expected_answer[:2]}' "
            "or point out exactly which part of the word is misspelled."
        )
    else:
        hint_depth = "Encourage them, they are about to see the answer."

    system_instruction = (
        f"You are an AI German Tutor for {difficulty} learners. "
        f"User input: '{user_input}'. Correct answer: '{expected_answer}'. "
        f"TASK: {hint_depth} "
        "CRITICAL: Do NOT reveal the full correct word. Keep the tip to 1 short sentence max."
    )

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[types.Content(role="user", parts=[types.Part(text=user_input)])],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "tip": types.Schema(type=types.Type.STRING),
                        },
                        required=["tip"]
                    )
                )
            )
        )
        return json.loads(response.text.strip())
    except Exception as e:
        # Simple fallback if AI fails or rate limit hit
        return {"tip": "Look closely at the spelling or the article!"}

async def generate_gec_task(user_id: str, difficulty: str, task_context: str) -> dict:
    """
    Optional: Kept if you want to generate general grammar challenges 
    outside of the main vocab experiment.
    """
    if not CLIENT_INITIALIZED:
        raise ConnectionError("Gemini API Client failed to initialize.")
        
    loop = asyncio.get_event_loop()
    
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[types.Content(role="user", parts=[types.Part(text=f"Topic: {task_context}")])],
                config=types.GenerateContentConfig(
                    system_instruction=f"Create a flawed German sentence at {difficulty} level and its correction.",
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "flawed_sentence": types.Schema(type=types.Type.STRING),
                            "expected_correction": types.Schema(type=types.Type.STRING),
                        },
                        required=["flawed_sentence", "expected_correction"]
                    )
                )
            )
        )
        return json.loads(response.text.strip())
    except Exception as e:
        raise Exception(f"Task generation failed: {str(e)}")
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
        return (f"You are a generator of German language challenges. Create ONE complex, full German sentence "
                f"at the {difficulty} level that contains a common grammatical error (e.g., incorrect case, "
                f"article, or verb placement). Return ONLY the flawed sentence and the corresponding correction.")
    
    elif task_type == 'VOCAB_CORRECTION':
        return (f"You are a specialized German vocabulary correction assistant for learners at the {difficulty} level. "
                f"Your task is to compare the user's input to the correct German vocabulary word and provide feedback. "
                f"Analyze the mistake (e.g., wrong article, wrong spelling). "
                f"Provide a concise 'corrected_text' (e.g., 'Incorrect article') and a detailed, helpful **tip in English** "
                f"in the 'tip' field explaining the specific mistake.")

    else:
        return (f"You are a high-precision German GEC (Grammatical Error Correction) engine. "
                f"The user is a learner at the {difficulty} level. Your task is to correct the user's text. "
                f"Provide ONLY the completely corrected sentence as the 'corrected_text' and a brief, encouraging feedback tip in English in the 'tip' field.")

def build_user_prompt(topic: str):
    """Builds the simple user prompt to guide the topic selection."""
    return f"Wähle {topic}-bezogene Wörter."


async def real_time_correction(user_input: str, expected_answer: str, attempt_number: int, difficulty: str, history: list = None) -> dict:
    """
    Generates specific pedagogical hints based on the attempt number.
    - Attempt 1: Subtle category/gender hint.
    - Attempt 2: Structural hint (e.g., first two letters).
    - Attempt 3+: Personalized pedagogical summary based on mistake history.
    """
    if not CLIENT_INITIALIZED:
        return {"tip": "Please review the gender and spelling of this noun."}

    # Format history for the AI to allow pattern analysis
    history_context = f" The learner made the following attempts: {', '.join(history)}." if history else ""

    # Differentiate logic based on attempt number to provide progressive scaffolding
    if attempt_number == 1:
        # Level 1: Subtle category focus
        task_desc = (
            f"The goal is '{expected_answer}'. User said '{user_input}'. "
            "Give a very subtle pedagogical hint in English (max 10 words). "
            "Focus on the gender category or word type. NEVER mention specific letters."
        )
    elif attempt_number == 2:
        # Level 2: Strong structural focus (First letters)
        start_hint = expected_answer[:2] if len(expected_answer) > 2 else expected_answer[0]
        task_desc = (
            f"The goal is '{expected_answer}'. User failed twice. User said '{user_input}'.{history_context} "
            f"Give a strong structural hint in English. Mention that the word starts with '{start_hint}' "
            "or point out exactly which part is misspelled."
        )
    else:
        # Level 3: Final Personalized Analysis
        task_desc = (
            f"The learner failed 3 times to get '{expected_answer}'. Mistake history: {history_context} "
            "Provide a personalized pedagogical summary in English (max 20 words). "
            "Analyze WHY their attempts were wrong (e.g., 'You seem to be confusing masculine and feminine articles'). "
            "Provide one clear tip for them to remember this next time."
        )

    system_instruction = (
        f"You are a professional AI German Tutor for {difficulty} level. "
        f"TASK: {task_desc} "
        f"CRITICAL: The entire response must be in English. Do not use German in the explanation. "
        f"Keep the response to exactly ONE short sentence."
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
                        properties={"tip": types.Schema(type=types.Type.STRING)},
                        required=["tip"]
                    )
                )
            )
        )
        return json.loads(response.text.strip())
    except Exception:
        return {"tip": "Review the patterns in your spelling and article gender choices."}

# --- Standard Task Generators ---

async def generate_vocab_task(user_id: str, num_items: int, topic: str) -> List[Dict]:
    """Calls the Gemini API to get structured vocabulary."""
    if not CLIENT_INITIALIZED:
        raise ConnectionError("Gemini API Client failed to initialize.")
        
    user_prompt = f"Choose {topic}-related words for A2 level."
    loop = asyncio.get_event_loop()
    
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[types.Content(role="user", parts=[types.Part(text=user_prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=f"Select exactly {num_items} German vocabulary words and translations. Return as JSON array.", 
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
        return json.loads(response.text.strip())
    except Exception as e:
        raise Exception(f"LLM Task Failure: {str(e)}")
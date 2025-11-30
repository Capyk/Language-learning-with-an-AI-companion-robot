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


async def generate_gec_task(user_id: str, difficulty: str, task_context: str) -> dict:
    """Calls the Gemini API to generate a challenging German sentence/prompt."""
    
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
                    system_instruction=build_gec_system_instruction(difficulty, 'GEC_CHALLENGE'),
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


async def real_time_correction(user_input: str, user_id: str, difficulty: str, expected_answer: str = None) -> dict:
    """
    Analyzes the user's input, compares it against an expected answer (if provided), 
    and provides correction and a detailed tip using Gemini.
    """
    
    if not CLIENT_INITIALIZED:
        raise ConnectionError("Gemini API Client failed to initialize.")

    # Determine the task type and build the prompt
    if expected_answer:
        # If expected_answer is present, this is a VOCAB_CORRECTION task
        task_type = 'VOCAB_CORRECTION'
        prompt = (f"User Input: '{user_input}'. Correct Answer: '{expected_answer}'. "
                  f"Analyze the mistake and generate the feedback required by the system instruction.")
    else:
        # Otherwise, this is a general GEC correction task
        task_type = 'CORRECTION'
        prompt = user_input
        
    loop = asyncio.get_event_loop()
    
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])], # Use the detailed prompt
                config=types.GenerateContentConfig(
                    system_instruction=build_gec_system_instruction(difficulty, task_type), # Use the correct instruction type
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "corrected_text": types.Schema(type=types.Type.STRING),
                            "tip": types.Schema(type=types.Type.STRING),
                        },
                        required=["corrected_text", "tip"]
                    )
                )
            )
        )
        # Check if the correction service successfully identified a mistake/provided a tip
        result = json.loads(response.text.strip())
        
        # If the user's input matches the expected answer, we override Gemini's GEC text with a simple "Correct!"
        if expected_answer and user_input.strip().lower() == expected_answer.strip().lower():
             return {"corrected_text": "Correct!", "tip": "Super gemacht! You nailed the German vocabulary."}

        return result
        
    except Exception as e:
        print(f"Gemini Correction Service Error: {e}")
        # Fallback response for failed correction service
        return {"corrected_text": f"Error processing input: {user_input}", "tip": f"Correction service failed: {str(e)}"}

# Keep the simple validation logic for the quiz phase separate
async def validate_answer(user_answer: str, correct_answer: str, user_id: str) -> Dict:
    """Simple validation logic for the quiz phase."""
    user_clean = user_answer.lower().strip()
    correct_clean = correct_answer.lower().strip()
    
    is_correct = user_clean == correct_clean
    
    return {
        "is_correct": is_correct,
        "feedback": "Korrekt! Wunderbar!" if is_correct else "Versuch es nochmal. Achte auf den Artikel!"
    }
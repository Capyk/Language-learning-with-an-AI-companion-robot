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
# ... existing code ...
        return json.loads(response.text.strip())
    except Exception as e:
        raise Exception(f"LLM Task Failure: {str(e)}")

async def generate_tutor_response(
    question: str, 
    context: Dict, 
    is_nudge: bool = False,
    target_language: str = "de"
) -> Dict:
    """
    Generates an A1-level friendly German tutor response.
    
    Args:
        question: User's question (or empty if it's a nudge).
        context: Context about the current task (prompt, user_answer, expected, is_correct).
        is_nudge: If True, generates a short unsolicited hint/correction without a direct user question.
        target_language: 'de' (German) or 'en' (English) - determines the language of the tutor's response.
    
    Returns:
        JSON dict with keys: message, correction, rule, mnemonic, example
    """
    if not CLIENT_INITIALIZED:
        return {
            "message": "Entschuldigung, ich kann gerade nicht antworten." if target_language == "de" else "Sorry, I cannot answer right now.",
            "correction": None, 
            "rule": None, 
            "mnemonic": None, 
            "example": None
        }

    # Context formatting
    task_prompt = context.get('prompt', 'Unbekannte Aufgabe')
    user_ans = context.get('user_answer', '(Keine Antwort)')
    expected = context.get('expected_answer', '')
    is_correct = context.get('is_correct')
    
    if target_language == 'en':
        # --- ENGLISH TUTOR PERSONA ---
        system_instruction = (
            "You are a friendly, patient German Tutor for beginners (A1 level). "
            "Your explanations must be in ENGLISH. "
            "Keep sentences simple and short. "
            "Avoid complex grammatical terms. "
            "ALWAYS answer in JSON format. "
            "Structure: \n"
            "- message: Your direct answer in English (max 1-2 sentences).\n"
            "- correction: Only if necessary, the correction of the user's German input (e.g., 'Correct is: ...'). Else null.\n"
            "- rule: A very short rule in English (max 1 sentence), if helpful. Else null.\n"
            "- mnemonic: A short mnemonic/memory aid in English, if possible. Else null.\n"
            "- example: A very simple German example (A1 level) that fits the explanation. Else null.\n\n"
            "Be empathetic. Mistakes are okay.\n"
        )
        
        if is_nudge:
             if is_correct:
                user_interaction = (
                    f"The user solved the task '{task_prompt}' correctly (Answer: '{user_ans}'). "
                    "Give short praise in English (e.g., 'Very good!' or 'Great job!') and maybe a tiny sentence."
                )
             else:
                 user_interaction = (
                    f"The user solved the task '{task_prompt}' incorrectly (Answer: '{user_ans}', Correct was: '{expected}'). "
                    "Give a short, encouraging hint in English. Mention the correct German solution and explain briefly why."
                )
        else:
            user_interaction = (
                f"Task Context: '{task_prompt}'. User Answer was: '{user_ans}' (Correct: {is_correct}, Expected: '{expected}').\n"
                f"The user asks you: \"{question}\"\n"
                "Answer the question in English contextually. Keep it simple."
            )

    else:
        # --- GERMAN TUTOR PERSONA (Default) ---
        system_instruction = (
            "Du bist ein freundlicher, geduldiger Deutsch-Tutor für Anfänger (Niveau A1). "
            "Deine Sprache ist einfach, deine Sätze sind kurz. "
            "Vermeide komplizierte grammatikalische Begriffe. Wenn nötig, erkläre sie sehr einfach. "
            "Antworte IMMER im JSON-Format. "
            "Struktur: \n"
            "- message: Deine direkte Antwort (max 1-2 Sätze).\n"
            "- correction: Nur falls nötig, die Korrektur des Nutzers (z.B. 'Richtig ist: ...'). Sonst null.\n"
            "- rule: Eine sehr kurze Regel (max 1 Satz), falls hilfreich. Sonst null.\n"
            "- mnemonic: Ein kurzer Merksatz (Eselsbrücke), falls möglich. Sonst null.\n"
            "- example: Ein sehr einfaches Beispiel (Niveau A1), das zur Erklärung passt. Sonst null.\n\n"
            "Verhalte dich empathisch. Fehler sind okay.\n"
        )

        if is_nudge:
            # Nudge Logic
            if is_correct:
                user_interaction = (
                    f"Der Nutzer hat die Aufgabe '{task_prompt}' richtig gelöst (Antwort: '{user_ans}'). "
                    "Gib ein sehr kurzes Lob (z.B. 'Sehr gut!' oder 'Klasse!') und vielleicht einen Minisatz dazu."
                )
            else:
                 user_interaction = (
                    f"Der Nutzer hat die Aufgabe '{task_prompt}' falsch gelöst (Antwort: '{user_ans}', Richtig wäre: '{expected}'). "
                    "Gib eine kurze, aufmunternde Hilfe. Nenne die korrekte Lösung und erkläre kurz warum (sehr einfach)."
                )
        else:
            # Direct Question Logic
            user_interaction = (
                f"Kontext Aufgabe: '{task_prompt}'. Nutzer-Antwort war: '{user_ans}' (Korrekt: {is_correct}, Richtig wäre: '{expected}').\n"
                f"Der Nutzer fragt dich: \"{question}\"\n"
                "Antworte auf die Frage im Kontext der Aufgabe. Bleib bei A1 Deutsch."
            )

    # Retry logic with exponential backoff
    max_retries = 3
    retry_delays = [0.5, 1.0, 2.0]  # seconds
    
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[types.Content(role="user", parts=[types.Part(text=user_interaction)])],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "message": types.Schema(type=types.Type.STRING),
                                "correction": types.Schema(type=types.Type.STRING, nullable=True),
                                "rule": types.Schema(type=types.Type.STRING, nullable=True),
                                "mnemonic": types.Schema(type=types.Type.STRING, nullable=True),
                                "example": types.Schema(type=types.Type.STRING, nullable=True)
                            },
                            required=["message"]
                        )
                    )
                )
            )
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(response.text.strip())
                # Ensure message field exists
                if "message" not in parsed_response or not parsed_response["message"]:
                    raise ValueError("Response missing 'message' field")
                return parsed_response
            except (json.JSONDecodeError, ValueError) as parse_error:
                print(f"[WARNING] Tutor JSON Parse Error (Attempt {attempt + 1}/{max_retries}): {parse_error}")
                print(f"[DEBUG] Raw response: {response.text[:200]}")
                
                # Fallback: Try to extract message with regex
                import re
                message_match = re.search(r'"message"\s*:\s*"([^"]+)"', response.text)
                if message_match:
                    print("[INFO] Extracted message via regex fallback")
                    return {
                        "message": message_match.group(1),
                        "correction": None,
                        "rule": None,
                        "mnemonic": None,
                        "example": None
                    }
                
                # If last attempt, raise the error
                if attempt == max_retries - 1:
                    raise
                    
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"[ERROR] Tutor LLM Error (Attempt {attempt + 1}/{max_retries}): {error_type}: {error_msg}")
            
            # Check if this is a retryable error
            retryable_errors = ["timeout", "rate", "503", "429", "connection", "network"]
            is_retryable = any(keyword in error_msg.lower() for keyword in retryable_errors)
            
            # If last attempt or non-retryable error, return error message
            if attempt == max_retries - 1 or not is_retryable:
                # Return specific error messages based on error type
                if "rate" in error_msg.lower() or "429" in error_msg:
                    msg = "Das Limit für Anfragen ist erreicht. Bitte warte einen Moment." if target_language == "de" else "The request limit has been reached. Please wait a moment."
                    return {
                        "message": msg,
                        "correction": None, "rule": None, "mnemonic": None, "example": None
                    }
                elif "timeout" in error_msg.lower():
                    msg = "Die Antwort dauert zu lange. Bitte versuche es nochmal." if target_language == "de" else "The response is taking too long. Please try again."
                    return {
                        "message": msg,
                        "correction": None, "rule": None, "mnemonic": None, "example": None
                    }
                else:
                    msg = "Entschuldigung, ich habe ein Problem. Bitte versuche es nochmal." if target_language == "de" else "Sorry, I'm having a problem. Please try again."
                    return {
                        "message": msg,
                        "correction": None, "rule": None, "mnemonic": None, "example": None
                    }
            
            # Wait before retry
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying in {retry_delays[attempt]} seconds...")
                await asyncio.sleep(retry_delays[attempt])
    
    # Fallback (should never reach here due to logic above)
    return {
        "message": "Entschuldigung, ich habe ein Problem. Bitte versuche es nochmal." if target_language == "de" else "Sorry, I'm having a problem. Please try again.",
        "correction": None, "rule": None, "mnemonic": None, "example": None
    }

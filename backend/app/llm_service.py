import asyncio
import random
import json
import logging
import time
import os
from typing import List, Dict
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM_SERVICE")

client = None 
CLIENT_INITIALIZED = False

TARGET_MODEL = "gemini-2.5-flash-preview-09-2025"

def _ensure_client():
    global client, CLIENT_INITIALIZED
    if CLIENT_INITIALIZED and client: return True
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY not found!")
        return False
    try:
        client = genai.Client(api_key=api_key)
        CLIENT_INITIALIZED = True
        logger.info("✅ Gemini Client initialized.")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini Init Error: {e}")
        return False

def clean_word_data(word: Dict, aggressive=True) -> Dict:
    """
    aggressive=True: usuwa ukośniki (do ćwiczeń) -> 'Besitzer'
    aggressive=False: zostawia ukośniki (do fiszek) -> 'Besitzer/Besitzerin'
    """
    clean = word.copy()
    raw_art = str(clean.get('article', ''))
    raw_word = str(clean.get('german_word', ''))
    
    if aggressive:
        if '/' in raw_art: clean['article'] = raw_art.split('/')[0].strip()
        if '/' in raw_word: clean['german_word'] = raw_word.split('/')[0].strip()
        if '/' in str(clean.get('plural', '')): clean['plural'] = str(clean['plural']).split('/')[0].strip()
    return clean

# --- STATIC GENERATOR (Group A) ---
def generate_static_learning_path_A(words: List[Dict]) -> List[Dict]:
    logger.info(">>> GENERATING STATIC PATH (GROUP A)")
    intro_screens = []
    presentation_screens = []
    practice_screens = []
    outro_screens = []
    
    intro_screens.append({
        "step_number": 0, "title": "Module Start",
        "content": "First, study all 5 words carefully.", "visual_type": "intro", "interaction_type": "read_only"
    })

    for i, word in enumerate(words):
        # FIX IV: Dla fiszki pełna nazwa, dla logiki skrócona
        display_data = clean_word_data(word, aggressive=False)
        logic_data = clean_word_data(word, aggressive=True)
        
        img_url = f"/images/{word.get('image_id')}.jpg"
        
        # 1. Presentation
        presentation_screens.append({
            "step_number": 0,
            "title": f"Learn: {display_data.get('english_gloss')}",
            "content": "Memorize the word, article and plural.",
            "visual_type": "word_card",
            "german_word": display_data.get('german_word'), # Shows 'Besitzer/Besitzerin'
            "article": display_data.get('article'),
            "plural": logic_data.get('plural'),
            "image_url": img_url,
            "example_sentence": word.get('example_de', ''), "interaction_type": "read_only"
        })

        # 2. Practice Spelling (Use Clean Logic Data for validation)
        practice_screens.append({
            "step_number": 0,
            "title": f"Practice: {display_data.get('english_gloss')}",
            "content": f"Type the word (Case Sensitive!):", "visual_type": "challenge",
            "image_url": img_url, 
            "question_context": word.get('example_de', '').replace(logic_data.get('german_word'), "_______"),
            "german_word": logic_data.get('german_word'), # Expects 'Besitzer'
            "interaction_type": "fill_gap"
        })

        # 3. Gender Check
        practice_screens.append({
            "step_number": 0,
            "title": f"Gender Check: {logic_data.get('german_word')}",
            "content": f"Select the correct article:", "visual_type": "challenge",
            "question_context": f"___ {logic_data.get('german_word')}",
            "german_word": f"{logic_data.get('article')} {logic_data.get('german_word')}", # "der Tisch"
            "interaction_type": "choice", "options": ["der", "die", "das"]
        })

    random.shuffle(practice_screens)
    
    outro_screens.append({"step_number": 0, "title": "Ready!", "content": "Starting the final test now.", "visual_type": "intro", "interaction_type": "read_only"})
    
    full_path = intro_screens + presentation_screens + practice_screens + outro_screens
    for idx, screen in enumerate(full_path): screen['step_number'] = idx + 1
    return full_path


# --- ADAPTIVE GENERATOR (Group B) ---
async def generate_adaptive_learning_path_B(words: List[Dict], error_logs: List[Dict]) -> List[Dict]:
    logger.info(">>> ATTEMPTING TO GENERATE ADAPTIVE AI PATH (GROUP B)")
    
    if not _ensure_client(): return generate_static_learning_path_A(words)

    # 1. Analyze Data
    performance_summary = []
    # Mapa do naprawy obrazków
    word_to_image_map = {}
    
    # Tworzymy czyste słowa do logiki, ale zachowujemy oryginały
    cleaned_logic_words = [clean_word_data(w, aggressive=True) for w in words]

    for w_orig, w_clean in zip(words, cleaned_logic_words):
        # Mapujemy "tisch" -> img_id
        word_to_image_map[w_clean['german_word'].lower()] = w_orig.get('image_id')
        
        german = w_clean.get('german_word')
        related_logs = [log for log in error_logs if german in str(log.get('word', ''))]
        
        # Check score (must be 1.0 to be correct)
        status = "CORRECT"
        if related_logs and not all(l.get('score', 0) == 1.0 for l in related_logs):
            status = "INCORRECT"
        
        performance_summary.append(f"- {german}: {status}")

    performance_str = "\n".join(performance_summary)
    
    words_metadata = []
    for w in words:
        # Pass full data so AI can see 'Besitzer/Besitzerin'
        words_metadata.append({
            "german": w.get('german_word'), "english": w.get('english_gloss'),
            "article": w.get('article'), "plural": w.get('plural'),
            "sentence": w.get('example_de'), "image_id": w.get('image_id')
        })

    # 2. PROMPT
    prompt = f"""
    You are an expert German Tutor. Create a personalized learning path (JSON).

    ### INPUT
    VOCAB: {json.dumps(words_metadata)}
    PERFORMANCE: {performance_str}

    ### INSTRUCTIONS
    Generate 'LearningScreen' objects.

    **STRATEGY:**
    1. **IF STATUS IS 'CORRECT':**
       - Generate 1 screen: 'word_card'. Content: "Quick review."

    2. **IF STATUS IS 'INCORRECT' (Reinforcement needed):**
       - Generate 3 screens:
       * **Screen A:** 'word_card'. Add 'mnemonics'.
       * **Screen B:** 'challenge' (fill_gap). Content: "Practice spelling.". Context: Sentence with gap. 'german_word' = Noun.
       * **Screen C:** 'challenge' (choice). Content: "Check article.". Context: "___ Noun". 
         **CRITICAL (Fix III):** Set 'german_word' to the FULL PHRASE (e.g. "der Tisch"), NOT just "der". The system needs the noun to find the image!

    3. **FINAL STEPS:**
       - Generate 3 distinct text screens: 'story', 'dialogue', 'fun_fact'.

    ### JSON FORMAT
    Return ONLY raw JSON.
    [
      {{
        "step_number": 0,
        "title": "Str", "content": "Str",
        "visual_type": "intro"|"word_card"|"story"|"challenge"|"dialogue"|"fun_fact",
        "interaction_type": "read_only"|"fill_gap"|"choice",
        "german_word": "Str", "article": "Str", "plural": "Str",
        "image_url": "PLACEHOLDER", 
        "example_sentence": "Str", "mnemonics": "Str",
        "question_context": "Str", "options": ["der","die","das"]
      }}
    ]
    """

    # 3. API CALL
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to Gemini (Model: {TARGET_MODEL})...")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=TARGET_MODEL,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4),
                    contents=[prompt]
                )
            )
            
            raw_text = response.text.strip()
            if raw_text.startswith("```json"): raw_text = raw_text[7:]
            if raw_text.endswith("```"): raw_text = raw_text[:-3]
            generated_path = json.loads(raw_text)
            
            # --- SAFETY LOCK (Fix Images III) ---
            for screen in generated_path:
                tgt = screen.get('german_word', '')
                if not tgt: continue
                
                # Jeśli mamy "der Tisch", wyciągamy "tisch" do szukania ID
                # Usuwamy rodzajniki
                clean_tgt = tgt.replace("der ", "").replace("die ", "").replace("das ", "").strip().lower()
                
                found_id = None
                if clean_tgt in word_to_image_map:
                    found_id = word_to_image_map[clean_tgt]
                else:
                    # Fuzzy search
                    for k, v in word_to_image_map.items():
                        if k in clean_tgt or clean_tgt in k:
                            found_id = v; break
                
                if found_id: 
                    screen['image_url'] = f"/images/{found_id}.jpg"
            # ------------------------------------

            final_path = [
                {"step_number": 0, "title": "AI Plan", "content": f"Focusing on {incorrect_words_count} items.", "visual_type": "intro", "interaction_type": "read_only"}
            ] + generated_path + [
                {"step_number": 0, "title": "Ready", "content": "Starting Final Test.", "visual_type": "intro", "interaction_type": "read_only"}
            ]

            for i, screen in enumerate(final_path): screen['step_number'] = i + 1
            return final_path

        except Exception as e:
            logger.error(f"❌ LLM ERROR: {e}")
            time.sleep(2)

    return generate_static_learning_path_A(words)

async def real_time_correction(user_input, expected, attempt, level, history):
    return {"tip": "Check spelling."}

async def generate_tutor_response(
    question: str, 
    context: Dict, 
    is_nudge: bool = False,
    target_language: str = "de"
) -> Dict:
    """
    Generates an A1-level friendly German tutor response.
    If is_nudge=True, it generates an unsolicited helpful tip based on the context.
    If is_nudge=False, it answers the user's specific question.
    """
    if not _ensure_client():
        return {
            "message": "Entschuldigung, ich kann gerade nie antworten." if target_language == "de" else "Sorry, I cannot answer right now.",
            "correction": None, "rule": None, "mnemonic": None, "example": None
        }

    # 1. Build the Persona and Prompt
    persona = "You are 'Lukas', a friendly and encouraging A1 German tutor for beginners."
    
    if is_nudge:
        prompt = f"""
        {persona}
        The student is practicing: '{context.get('prompt')}'
        Student's answer: '{context.get('user_answer')}'
        Expected answer: '{context.get('expected_answer')}'
        Is it correct? {context.get('is_correct')}

        TASK: Provide a very brief, helpful, and encouraging nudge in German (with English translation if complex).
        Focus on the specific error if any, or provide a related fun fact/mnemonic if they were correct.
        Keep it to 1-2 sentences.
        """
    else:
        prompt = f"""
        {persona}
        Current context: Student is practicing '{context.get('prompt')}' (Target: '{context.get('expected_answer')}')
        Student's question: "{question}"

        TASK: Answer the question simply. Use A1 German where possible.
        Return structured feedback.
        """

    # 2. Call Gemini
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=TARGET_MODEL,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
                contents=[prompt + "\nFormat: { 'message': '...', 'correction': '...', 'rule': '...', 'mnemonic': '...', 'example': '...' }"]
            )
        )
        
        raw_text = response.text.strip()
        # Clean potential markdown wrapping
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]
        
        data = json.loads(raw_text)
        
        # Translate message if target language is English
        if target_language == "en" and data.get("message"):
            # Simple fallback for this demo, usually you'd call the LLM again or have it dual-output
            data["message"] = data["message"] # LLM typically follows target_language if specified in prompt
            
        return data

    except Exception as e:
        logger.error(f"❌ Tutor LLM Error: {e}")
        return {
            "message": "Ich habe gerade technische Probleme, aber mach weiter! Du schaffst das.",
            "correction": None, "rule": None, "mnemonic": None, "example": None
        }
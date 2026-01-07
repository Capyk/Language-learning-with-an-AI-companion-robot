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

def clean_word_data(word: Dict) -> Dict:
    clean = word.copy()
    if '/' in str(clean.get('article', '')):
        clean['article'] = str(clean['article']).split('/')[0].strip()
    if '/' in str(clean.get('german_word', '')):
        clean['german_word'] = str(clean['german_word']).split('/')[0].strip()
    if '/' in str(clean.get('plural', '')):
        clean['plural'] = str(clean['plural']).split('/')[0].strip()
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

    cleaned_words = [clean_word_data(w) for w in words]

    for i, word in enumerate(cleaned_words):
        img_url = f"/images/{word.get('image_id')}.jpg"
        german = word.get('german_word', '')
        article = word.get('article', '')
        plural = word.get('plural', '')
        english = word.get('english_gloss', '')
        sentence = word.get('example_de', '')
        
        presentation_screens.append({
            "step_number": 0, "title": f"Learn: {english}",
            "content": "Memorize the word, article and plural.", "visual_type": "word_card",
            "german_word": german, "article": article, "plural": plural, "image_url": img_url,
            "example_sentence": sentence, "interaction_type": "read_only"
        })

        practice_screens.append({
            "step_number": 0, "title": f"Practice: {english}",
            "content": f"Type the missing word exactly:", "visual_type": "challenge",
            "image_url": img_url, "question_context": sentence.replace(german, "_______"),
            "german_word": german, "interaction_type": "fill_gap"
        })

        practice_screens.append({
            "step_number": 0, "title": f"Gender Check: {german}",
            "content": f"Select the correct article:", "visual_type": "challenge",
            "question_context": f"___ {german}",
            "german_word": f"{article} {german}", # Pełne słowo dla spójności
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
    
    if not _ensure_client():
        return generate_static_learning_path_A(words)

    # 1. Analyze Data
    performance_summary = []
    cleaned_words = [clean_word_data(w) for w in words]
    incorrect_words_count = 0
    word_to_image_map = {w['german_word']: w['image_id'] for w in cleaned_words}

    for word_data in cleaned_words:
        german = word_data.get('german_word')
        related_logs = [log for log in error_logs if german in str(log.get('word', ''))]
        status = "CORRECT"
        if related_logs and not all(log.get('is_correct') for log in related_logs):
            status = "INCORRECT"
            incorrect_words_count += 1
        performance_summary.append(f"- {german}: {status}")

    performance_str = "\n".join(performance_summary)
    words_metadata = []
    for w in cleaned_words:
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
    Generate a JSON array of 'LearningScreen' objects.

    **LOGIC:**
    
    1. **IF STATUS IS 'CORRECT':**
       - Generate ONLY 1 screen: 'word_card'.
       - Content: "You already know this! Quick review."

    2. **IF STATUS IS 'INCORRECT' (Reinforcement needed):**
       - Generate 3 screens for this word:
       
       * **Screen A:** 'word_card'. 
         - ADD 'mnemonics': Create a vivid memory hook using colors (Blue=Der, Red=Die, Green=Das).

       * **Screen B:** 'challenge' (fill_gap).
         - Content: "Practice the spelling."
         - Question Context: Use the example sentence, replace word with '_______'.
         - 'german_word': The noun itself (e.g. "Tisch").

       * **Screen C:** 'challenge' (choice).
         - Content: "Check the article."
         - Question Context: "___ Noun"
         - 'german_word': Full noun with article (e.g. "der Tisch").
         - Options: ["der", "die", "das"].

    3. **FINAL STEP:**
       - Generate 3 distinct text screens to conclude:
       * **A:** 'story' (Short funny story with tricky words).
       * **B:** 'dialogue' (Short conversation using the words).
       * **C:** 'fun_fact' (Interesting cultural fact about one word).

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
            logger.info(f"Sending request to Gemini (Model: {TARGET_MODEL}, Attempt {attempt + 1})...")
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=TARGET_MODEL,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.4
                    ),
                    contents=[prompt]
                )
            )
            
            raw_text = response.text.strip()
            if raw_text.startswith("```json"): raw_text = raw_text[7:]
            if raw_text.endswith("```"): raw_text = raw_text[:-3]
            
            generated_path = json.loads(raw_text)
            
            # --- SAFETY LOCK ---
            for screen in generated_path:
                tgt_word = screen.get('german_word')
                # Safety for choice: if AI put just "der", find the noun
                if tgt_word in ["der", "die", "das"]:
                     for k, v in word_to_image_map.items():
                        if k in str(screen.get('question_context', '')):
                            tgt_word = k
                            break
                if tgt_word:
                    # Case insensitive match
                    img_id = word_to_image_map.get(tgt_word)
                    if not img_id:
                        for k, v in word_to_image_map.items():
                            if k.lower() == tgt_word.lower(): img_id = v; break
                    if img_id: screen['image_url'] = f"/images/{img_id}.jpg"
            # -------------------

            final_path = [
                {"step_number": 0, "title": "AI ADAPTIVE PLAN", "content": f"I focused on your {incorrect_words_count} mistakes.", "visual_type": "intro", "interaction_type": "read_only"}
            ] + generated_path + [
                {"step_number": 0, "title": "AI Session Complete", "content": "Ready for final test.", "visual_type": "intro", "interaction_type": "read_only"}
            ]

            for i, screen in enumerate(final_path): screen['step_number'] = i + 1
            logger.info("✅ Adaptive path generated successfully.")
            return final_path

        except Exception as e:
            logger.error(f"❌ LLM ERROR (Attempt {attempt+1}): {e}")
            time.sleep(2)

    return generate_static_learning_path_A(words)

async def real_time_correction(user_input, expected, attempt, level, history):
    return {"tip": "Check spelling."}
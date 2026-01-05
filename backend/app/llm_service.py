import asyncio
import random
import json
import logging
from typing import List, Dict
from google import genai
from google.genai import types

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = None 
CLIENT_INITIALIZED = False

# --- STATIC GENERATOR (Group A) ---
# (Zostawiamy bez zmian, jak w poprzedniej wersji)
def generate_static_learning_path_A(words: List[Dict]) -> List[Dict]:
    intro_screens = []
    presentation_screens = []
    practice_screens = []
    outro_screens = []
    
    # 1. Intro
    intro_screens.append({
        "step_number": 0,
        "title": "Module Start",
        "content": "First, study all 5 words carefully. Then, we will shuffle the exercises to test your memory.",
        "visual_type": "intro",
        "interaction_type": "read_only"
    })

    # Loop through words to create screens
    for i, word in enumerate(words):
        img_url = f"/images/{word.get('image_id')}.jpg"
        german = word.get('german_word', '')
        article = word.get('article', '')
        plural = word.get('plural', '')
        english = word.get('english_gloss', '')
        sentence = word.get('example_de', '')
        
        # --- PHASE 1: Presentation ---
        presentation_screens.append({
            "step_number": 0,
            "title": f"Learn: {english}",
            "content": "Memorize the word, article and plural.",
            "visual_type": "word_card",
            "german_word": german,
            "article": article,
            "plural": plural,
            "image_url": img_url,
            "example_sentence": sentence,
            "interaction_type": "read_only"
        })

        # --- PHASE 2: Practice Items ---
        
        # A. Spelling Check
        context_sentence = sentence.replace(german, "_______")
        practice_screens.append({
            "step_number": 0,
            "title": f"Practice: {english}",
            "content": f"Type the missing word exactly (Watch out for Capital letters!):",
            "visual_type": "challenge",
            "image_url": img_url,
            "question_context": context_sentence,
            "german_word": german,
            "interaction_type": "fill_gap"
        })

        # B. Gender Check
        practice_screens.append({
            "step_number": 0,
            "title": f"Gender Check: {german}",
            "content": f"Select the correct article for '{german}':",
            "visual_type": "challenge",
            "question_context": f"___ {german}",
            "german_word": article, 
            "interaction_type": "choice",
            "options": ["der", "die", "das"]
        })

    random.shuffle(practice_screens)

    outro_screens.append({
        "step_number": 0,
        "title": "Ready!",
        "content": "Great job! You have practiced all words. Starting the final test now.",
        "visual_type": "intro",
        "interaction_type": "read_only"
    })
    
    full_path = intro_screens + presentation_screens + practice_screens + outro_screens
    
    for idx, screen in enumerate(full_path):
        screen['step_number'] = idx + 1

    return full_path


# --- ADAPTIVE GENERATOR (Group B - AI POWERED) ---
async def generate_adaptive_learning_path_B(words: List[Dict], error_logs: List[Dict]) -> List[Dict]:
    """
    Uses Gemini to generate a personalized learning path based on pre-test errors.
    """
    # 1. Safety Check: If API not ready, fallback to Static
    if not CLIENT_INITIALIZED:
        print("⚠️ LLM not initialized. Falling back to Static Path A.")
        return generate_static_learning_path_A(words)

    # 2. Analyze User Performance
    performance_summary = []
    incorrect_words = []
    
    for word_data in words:
        german = word_data.get('german_word')
        # Find logs related to this word
        related_logs = [log for log in error_logs if log.get('word') == german]
        
        if not related_logs:
            # No data (shouldn't happen if pre-test covers all), assume correct
            status = "correct"
        else:
            # Check if any attempt was wrong
            is_correct = all(log.get('is_correct') for log in related_logs)
            if is_correct:
                status = "correct"
            else:
                status = "incorrect"
                incorrect_words.append(german)
                # Analyze specific error (e.g., used wrong article)
                last_input = related_logs[-1].get('user_input', '')
                status += f" (User typed: '{last_input}')"

        performance_summary.append(f"- {german}: {status}")

    performance_str = "\n".join(performance_summary)
    
    # 3. Prepare Prompt Data
    # We pass the full metadata so LLM can use it (images, sentences)
    words_metadata = []
    for w in words:
        words_metadata.append({
            "german": w.get('german_word'),
            "english": w.get('english_gloss'),
            "article": w.get('article'),
            "plural": w.get('plural'),
            "sentence": w.get('example_de'),
            "image_id": w.get('image_id')
        })

    # 4. Construct the Prompt
    prompt = f"""
    You are an expert Adaptive German Tutor. Design a specific learning sequence (JSON) for a student based on their Pre-Test results.

    ### INPUT DATA
    TARGET WORDS: {json.dumps(words_metadata)}
    
    USER PRE-TEST PERFORMANCE:
    {performance_str}

    ### INSTRUCTIONS
    Generate a JSON array of "LearningScreen" objects. The path should adapt based on errors.
    
    **Principles:**
    1. **For Correct Words:** Provide a QUICK review (only 1 screen per word: 'word_card'). Do NOT add exercises.
    2. **For Incorrect Words:** Provide a DEEP review (3 screens per word):
       - Screen A: 'word_card' WITH A MNEMONIC in the 'mnemonics' field.
         * If the error was gender (e.g. wrong article), the mnemonic MUST use colors/imagery (Blue=Der, Red=Die, Green=Das).
         * Example: "Imagine the Table (Tisch) is made of BLUE ice (Der)."
       - Screen B: 'challenge' (fill_gap) to practice spelling.
       - Screen C: 'challenge' (choice) to practice the article.
    3. **The Connector:** After individual words, add a 'story' screen. Create a short, funny German text (2-3 sentences) that combines the words the user got WRONG.
    4. **Intro/Outro:** Add a brief 'intro' and 'intro' (as ready screen) at the end.

    ### JSON STRUCTURE (Strict)
    Return ONLY a raw JSON array. Each object must have:
    - "title": (string)
    - "content": (string, instructions)
    - "visual_type": "intro" | "word_card" | "story" | "challenge"
    - "interaction_type": "read_only" | "fill_gap" | "choice"
    - "german_word": (string, required for challenges/cards)
    - "article": (string, required for cards)
    - "plural": (string, required for cards)
    - "image_url": (string, format: "/images/IMAGE_ID.jpg" - use the provided image_ids!)
    - "example_sentence": (string)
    - "mnemonics": (string, optional, for adaptive hints)
    - "question_context": (string, for challenges. For fill_gap, use '_______' as placeholder)
    - "options": (array of strings, only for 'choice' interaction)

    DO NOT wrap in markdown code blocks. Just valid JSON.
    """

    # 5. Call Gemini
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model='gemini-2.0-flash', 
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4  # Lower temp for more structured output
                ),
                contents=[prompt]
            )
        )
        
        # 6. Parse and Post-process
        raw_text = response.text.strip()
        # Remove potential markdown formatting if Gemini adds it despite instructions
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]
        
        generated_path = json.loads(raw_text)
        
        # Add step numbers
        for i, screen in enumerate(generated_path):
            screen['step_number'] = i + 1
            
        return generated_path

    except Exception as e:
        logger.error(f"LLM Generation Failed: {e}")
        # Fallback to static path if LLM crashes
        return generate_static_learning_path_A(words)

# --- HELPER (Legacy) ---
async def real_time_correction(user_input, expected, attempt, level, history):
    return {"tip": "Check your article and spelling."}
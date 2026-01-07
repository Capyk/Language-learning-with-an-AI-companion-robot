import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv() # Ładuje klucz z .env

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Dostępne modele dla Twojego klucza:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name.replace('models/', '')}")
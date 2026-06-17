# config_ai.py
import os

# PILIHAN: 'GEMINI' atau 'OPENAI'
ACTIVE_AI = os.getenv("ACTIVE_AI", "GEMINI") 

# Load API Keys dari .env
from api_keys import get_gemini_api_key, get_openai_api_key

GEMINI_KEY = get_gemini_api_key()
OPENAI_KEY = get_openai_api_key()

if ACTIVE_AI == "GEMINI" and not GEMINI_KEY:
    print("[WARNING] ACTIVE_AI is GEMINI but no API key found!")
elif ACTIVE_AI == "OPENAI" and not OPENAI_KEY:
    print("[WARNING] ACTIVE_AI is OPENAI but no API key found!")
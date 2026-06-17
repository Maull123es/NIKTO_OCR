# api_keys.py
import os
from dotenv import load_dotenv

# Load variabel dari file .env
load_dotenv()

def get_gemini_api_key():
    """Mengambil API Key Gemini dari environment."""
    return os.getenv("GEMINI_API_KEY")

def get_openai_api_key():
    """Mengambil API Key OpenAI (GPT) dari environment."""
    return os.getenv("OPENAI_API_KEY")
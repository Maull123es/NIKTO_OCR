# ai_engine.py
import os
import json
import tempfile
import cv2
from dotenv import load_dotenv

load_dotenv()

class AIEngine:
    def __init__(self):
        self.backend = os.getenv("AI_BACKEND", "local").lower()
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
        
        # Inisialisasi backend lokal (Ollama)
        try:
            from local_ocr import LocalOCR
            self.local_ocr = LocalOCR(model_name=self.model)
            print(f"[AI_ENGINE] ✅ Backend Lokal siap: {self.model}")
        except Exception as e:
            print(f"[AI_ENGINE] ❌ Gagal init LocalOCR: {e}")
            self.local_ocr = None

        # Placeholder untuk backend cloud (fallback safety net)
        self.gemini_ocr = None
        self.openai_ocr = None
        
        # Coba load cloud OCR kalau API key tersedia
        if os.getenv("GEMINI_API_KEY"):
            try:
                from gemini_ocr import GeminiOCR
                self.gemini_ocr = GeminiOCR()
                print("[AI_ENGINE] ✅ Backend Gemini siap")
            except Exception as e: 
                print(f"[AI_ENGINE] ❌ Gagal memuat GeminiOCR: {e}")
            
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai_ocr import OpenAIOCR
                self.openai_ocr = OpenAIOCR()
                print("[AI_ENGINE] ✅ Backend OpenAI siap")
            except Exception as e: 
                print(f"[AI_ENGINE] ❌ Gagal memuat OpenAIOCR: {e}")

    def scan_ljk(self, image_path: str):
        if self.backend == 'local' and self.local_ocr:
            return self.local_ocr.scan_ljk(image_path)
        elif self.backend == 'gemini' and self.gemini_ocr:
            return self.gemini_ocr.scan_ljk(image_path)
        elif self.backend == 'openai' and self.openai_ocr:
            return self.openai_ocr.scan_ljk(image_path)
        else:
            return {"error": "Backend AI tidak tersedia atau API key tidak diset"}

# Singleton pattern agar model tidak diload berulang
_engine_instance = None
def get_ai_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AIEngine()
    return _engine_instance

def tebak_huruf_ai(img_array):
    """
    Fungsi pembantu untuk kompatibilitas dengan ocr.py lama.
    Menerima numpy array dari gambar kotak jawaban.
    """
    temp_path = None
    try:
        # PATH DINAMIS: Menyimpan di folder 'uploads' dalam direktori kerja saat ini
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(suffix='.jpg', dir=upload_dir)
        os.close(fd)
        
        cv2.imwrite(temp_path, img_array)

        engine = get_ai_engine()
        result = engine.scan_ljk(temp_path)
        
        if "error" in result or not result.get("jawaban_siswa"):
            return "", 0.0
            
        huruf = str(result["jawaban_siswa"][0]).strip().upper()
        
        if huruf in ['A', 'B', 'C', 'D', 'E']:
            return huruf, 0.95
        return "", 0.0
        
    except Exception as e:
        print(f"[COMPAT WRAPPER] Error processing crop: {e}")
        return "", 0.0
        
    finally:
        # Pindahkan penghapusan ke blok finally agar selalu dieksekusi
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
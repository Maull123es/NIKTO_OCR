# gemini_ocr.py
from google import genai
from google.genai import types
import io
import time
import cv2
from PIL import Image
from api_keys import get_gemini_api_key

GEMINI_API_KEY = get_gemini_api_key()
request_counter = 0

def tebak_pake_gemini(img_crop):
    global request_counter
    
    if not GEMINI_API_KEY:
        return "", 0.0
        
    try:
        request_counter += 1
        
        if request_counter % 5 == 0:
            print(f"[RATE LIMIT] Istirahat 3 detik... (Request #{request_counter})")
            time.sleep(3)
            
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Konversi warna jika gambar memiliki 3 channel (BGR to RGB)
        if len(img_crop.shape) == 3:
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            
        pil_image = Image.fromarray(img_crop)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()
        
        prompt = "Identifikasi huruf tulisan tangan pada gambar ini. Jawab HANYA dengan satu huruf: A, B, C, D, atau E."
        
        response = client.models.generate_content(
            model='gemini-1.5-flash', # SDK baru lebih stabil tanpa -latest
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
            ]
        )
        
        jawaban = response.text.strip().upper()
        
        valid_chars = ['A', 'B', 'C', 'D', 'E']
        for char in jawaban:
            if char in valid_chars:
                return char, 1.0
                
        return "", 0.0
            
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("[WARN]  Kena Quota Limit Gemini! Istirahat 60 detik...")
            time.sleep(60)
        return "", 0.0

# === CLASS WRAPPER UNTUK AI ENGINE ===
class GeminiOCR:
    def scan_ljk(self, image_path: str) -> dict:
        img_crop = cv2.imread(image_path)
        if img_crop is None:
            return {"error": "Gagal membaca gambar fallback"}
            
        huruf, conf = tebak_pake_gemini(img_crop)
        if conf > 0:
            return {"jawaban_siswa": [huruf]}
        return {"error": "Gemini gagal mendeteksi huruf"}
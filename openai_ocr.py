# openai_ocr.py
import base64
import io
import time
import cv2
from PIL import Image
from openai import OpenAI
from api_keys import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

def tebak_pake_openai(img_crop):
    if not OPENAI_API_KEY:
        return "", 0.0
        
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        if len(img_crop.shape) == 3:
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            
        pil_image = Image.fromarray(img_crop)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        prompt_text = (
            "You are an expert in reading handwritten multiple-choice answer sheets. "
            "The image contains a single handwritten letter inside a box. "
            "Identify the letter strictly as one of these options: A, B, C, D, or E. "
            "Respond with ONLY the single uppercase letter. Do not add any explanation."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=5
        )
        
        jawaban = response.choices[0].message.content.strip().upper()
        
        if jawaban in ['A', 'B', 'C', 'D', 'E']:
            return jawaban, 1.0
        else:
            for char in jawaban:
                if char in ['A', 'B', 'C', 'D', 'E']:
                    return char, 0.9
            return "", 0.0
            
    except Exception as e:
        print(f"[ERROR OPENAI] {e}")
        # Rate limit handling dinamis
        if "429" in str(e):
            print("[WARN] Kena Rate Limit OpenAI. Tidur 5 detik...")
            time.sleep(5)
        return "", 0.0

# === CLASS WRAPPER UNTUK AI ENGINE ===
class OpenAIOCR:
    def scan_ljk(self, image_path: str) -> dict:
        img_crop = cv2.imread(image_path)
        if img_crop is None:
            return {"error": "Gagal membaca gambar fallback"}
            
        huruf, conf = tebak_pake_openai(img_crop)
        if conf > 0:
            return {"jawaban_siswa": [huruf]}
        return {"error": "OpenAI gagal mendeteksi huruf"}
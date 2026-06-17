# local_ocr.py
import ollama
import os
import json
import re

class LocalOCR:
    def __init__(self, model_name="qwen2.5vl:3b"):
        self.model = model_name
        try:
            ollama.chat(model=self.model, messages=[{'role': 'user', 'content': 'test'}])
        except Exception as e:
            print(f"[LocalOCR] Warning: Gagal connect ke Ollama. Pastikan 'ollama serve' jalan. Error: {e}")

    def scan_ljk(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            return {"error": f"File tidak ditemukan: {image_path}"}

        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()

            prompt = """
            Kamu adalah sistem OCR khusus untuk Lembar Jawab Komputer (LJK). 
            Tugasmu adalah membaca jawaban siswa dari foto LJK yang diberikan.
            
            Analisis gambar dan kembalikan HASIL SAJA dalam format JSON berikut:
            {
                "nomor_soal": [1, 2, 3],
                "jawaban_siswa": ["A", "B", "C"],
                "kualitas_gambar": "baik/sedang/buruk"
            }
            """

            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }],
                options={
                    "temperature": 0.1,
                    "num_ctx": 4096
                }
            )

            result_text = response['message']['content'].strip()
            
            # PARSING JSON MENGGUNAKAN REGEX (Lebih kebal error)
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                raise json.JSONDecodeError("Blok JSON tidak ditemukan", result_text, 0)

        except json.JSONDecodeError as e:
            return {
                "error": "Gagal parse JSON dari AI",
                "raw_response": result_text[:200] if 'result_text' in locals() else "None",
                "detail": str(e)
            }
        except Exception as e:
            return {"error": f"Error saat processing: {str(e)}"}
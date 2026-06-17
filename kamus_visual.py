"""
kamus_visual.py — Visual Dictionary Matcher
Versi final: returns (huruf, confidence) tuple biar bisa ikut voting berbobot
"""

import cv2
import os
import numpy as np


def tebak_pake_kamus(crop_jawaban_siswa, folder_kamus="kamus_huruf"):
    """
    Return: (huruf_str, confidence_float)
    huruf_str  = 'A'–'E' atau '' kalau gagal
    confidence = 0.0–1.0
    """
    skor_terbaik  = 0.0
    huruf_terpilih = ""

    if not os.path.exists(folder_kamus) or crop_jawaban_siswa is None or crop_jawaban_siswa.size == 0:
        return "", 0.0

    try:
        img = crop_jawaban_siswa.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, img_biner = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_siswa   = cv2.resize(img_biner, (50, 50))

        for file_nama in os.listdir(folder_kamus):
            if not file_nama.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            huruf_target = file_nama[0].upper()
            if huruf_target not in ['A', 'B', 'C', 'D', 'E']:
                continue

            try:
                path_contoh   = os.path.join(folder_kamus, file_nama)
                img_contoh    = cv2.imread(path_contoh, cv2.IMREAD_GRAYSCALE)
                if img_contoh is None or img_contoh.size == 0:
                    continue

                _, img_contoh_biner = cv2.threshold(img_contoh, 127, 255, cv2.THRESH_BINARY)
                img_contoh_biner    = cv2.resize(img_contoh_biner, (50, 50))

                result = cv2.matchTemplate(img_siswa, img_contoh_biner, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > skor_terbaik:
                    skor_terbaik   = max_val
                    huruf_terpilih = huruf_target

            except Exception:
                continue

        if skor_terbaik >= 0.55:
            print(f"[KAMUS] Match '{huruf_terpilih}' conf={skor_terbaik:.2f}")
            return huruf_terpilih, round(float(skor_terbaik), 3)

    except Exception as e:
        print(f"[ERROR KAMUS]: {e}")

    return "", 0.0

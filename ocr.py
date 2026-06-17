"""
    _   ___ __   __
   / | / (_) /__/ /_____      ____  ____ ___  _____
  /  |/ / / //_/ __/ __ \    / __ \/ __ `__ \/ ___/
 / /|  / / ,< / /_/ /_/ /   / /_/ / / / / / / /
/_/ |_/_/_/|_|\__/\____/____\____/_/ /_/ /_/_/
                      /_____/
"""

import os
import cv2
import numpy as np
import json
import shutil
from datetime import datetime
from collections import Counter
from flask import Flask, request, jsonify, Response

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("[WARNING] pytesseract tidak terinstal. Fallback ke AI Engine saja.")

from kamus_visual import tebak_pake_kamus

# ─────────────────────────────── BANNER ─────────────────────────────────────
def print_banner():
    C_BLUE  = "\033[94m"
    C_RESET = "\033[0m"
    print(f"""
{C_BLUE}    _   ___ __   __
   / | / (_) /__/ /_____      ____  ____ ___  _____
  /  |/ / / //_/ __/ __ \\    / __ \\/ __ `__ \\/ ___/
 / /|  / / ,< / /_/ /_/ /   / /_/ / / / / / / /
/_/ |_/_/_/|_|\\__/\\____/____\\____/_/ /_/ /_/_/
                      /_____/{C_RESET}
""")

print_banner()

# ─────────────────────────────── INIT ────────────────────────────────────────
app = Flask(__name__)

# Aktifkan kalau Tesseract tidak ditemukan otomatis:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

OS_DEBUG_DIR = "debug_crop"
if os.path.exists(OS_DEBUG_DIR):
    shutil.rmtree(OS_DEBUG_DIR)
os.makedirs(OS_DEBUG_DIR, exist_ok=True)


# ─────────────────────────── KUNCI JAWABAN ───────────────────────────────────
def load_kunci_jawaban(filepath):
    kunci = {}
    if not os.path.exists(filepath):
        print(f"[PERINGATAN] File kunci tidak ditemukan: {filepath}")
        return kunci
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            try:
                no, ans = line.split(',', 1)
                no_clean  = int(no.strip())
                ans_clean = ans.strip().upper()
                if ans_clean in ['A', 'B', 'C', 'D', 'E']:
                    kunci[no_clean] = ans_clean
            except Exception:
                continue
    return kunci


# ──────────────────────────── DESKEW (SUPER ROBUST) ─────────────────────────
def luruskan_kertas(img):
    """
    Versi Super Robust:
    Selalu kembalikan gambar (grayscale/warped). Tidak pernah return None.
    """
    h_orig, w_orig = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return gray

    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    pts = approx.reshape(-1, 2).astype("float32")

    if len(pts) != 4:
        hull = cv2.convexHull(c)
        hull_pts = hull.reshape(-1, 2).astype("float32")
        if len(hull_pts) < 4:
            return gray 
        
        s = hull_pts.sum(axis=1)
        diff = np.diff(hull_pts, axis=1).ravel()
        tl = hull_pts[np.argmin(s)]
        br = hull_pts[np.argmax(s)]
        tr = hull_pts[np.argmin(diff)]
        bl = hull_pts[np.argmax(diff)]
        pts = np.array([tl, tr, br, bl], dtype="float32")

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    width_a = np.linalg.norm(rect[0] - rect[1])
    width_b = np.linalg.norm(rect[2] - rect[3])
    height_a = np.linalg.norm(rect[0] - rect[3])
    height_b = np.linalg.norm(rect[1] - rect[2])
    
    maxWidth = max(int(width_a), int(width_b))
    maxHeight = max(int(height_a), int(height_b))

    if maxWidth < 100 or maxHeight < 100:
        return gray

    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1],
    ], dtype="float32")

    try:
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped
    except Exception:
        return gray


# ──────────────────────── DETEKSI GRID HYBRID ────────────────────────────────
def dapatkan_kotak_hybrid(img_gray, total_soal=50):
    """
    Strategi Table-Based: Deteksi garis tabel untuk menemukan kotak jawaban.
    Ini jauh lebih akurat daripada pembagian rata jika struktur tabelnya jelas.
    """
    h, w = img_gray.shape
    
    # 1. Preprocessing untuk deteksi garis
    # Blur sedikit untuk menghilangkan noise pensil
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # Thresholding Otsu Invers (Hitam jadi Putih, Putih jadi Hitam)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Deteksi Garis Vertikal & Horizontal
    # Panjang kernel disesuaikan dengan ukuran gambar (estimasi)
    vertical_kernel_len = int(h * 0.5) # Garis vertikal panjangnya setengah tinggi gambar
    horizontal_kernel_len = int(w * 0.5) # Garis horizontal panjangnya setengah lebar gambar
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
    
    # Erosi untuk menyisakan hanya garis
    vertical_lines = cv2.erode(thresh, vertical_kernel, iterations=1)
    horizontal_lines = cv2.erode(thresh, horizontal_kernel, iterations=1)
    
    # Gabungkan garis untuk mendapatkan struktur tabel
    table_structure = cv2.add(vertical_lines, horizontal_lines)
    
    # Cari kontur dari struktur tabel
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ambil kontur terbesar (seharusnya itu tabel jawabannya)
    if not contours:
        print("[WARNING] Tidak ada struktur tabel terdeteksi. Fallback ke Regular Grid.")
        return dapatkan_kotak_regular_grid(img_gray, total_soal)
        
    # Filter kontur: Ambil yang luasnya signifikan (misal > 20% area gambar)
    valid_contours = [c for c in contours if cv2.contourArea(c) > (h*w*0.2)]
    if not valid_contours:
         print("[WARNING] Kontur tabel terlalu kecil. Fallback ke Regular Grid.")
         return dapatkan_kotak_regular_grid(img_gray, total_soal)
         
    table_contour = max(valid_contours, key=cv2.contourArea)
    x_tbl, y_tbl, w_tbl, h_tbl = cv2.boundingRect(table_contour)
    
    # Crop area tabel saja untuk analisis lebih lanjut
    roi_table = thresh[y_tbl:y_tbl+h_tbl, x_tbl:x_tbl+w_tbl]
    
    # 3. Deteksi Sel-sel Kecil di dalam Tabel
    # Kita cari kotak-kotak kecil di dalam area tabel tersebut
    # Gunakan morphological closing untuk menutup celah antar sel
    kernel_cell = np.ones((5,5), np.uint8)
    closed_cells = cv2.morphologyEx(roi_table, cv2.MORPH_CLOSE, kernel_cell)
    
    # Inversi lagi biar sel jadi putih di background hitam (untuk findContours)
    cells_inv = cv2.bitwise_not(closed_cells)
    
    # Cari kontur sel
    cell_contours, _ = cv2.findContours(cells_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    calon_kotak = []
    for cnt in cell_contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        # Filter sel yang ukurannya masuk akal untuk kotak jawaban
        # Asumsi: Lebar sel sekitar 1-5% dari lebar tabel, Tinggi sel sekitar 1-5% dari tinggi tabel
        if 0.01 < (cw/w_tbl) < 0.1 and 0.01 < (ch/h_tbl) < 0.15:
            # Pastikan sel berada di area kanan kolom (tempat jawaban A-E)
            # Biasanya kolom nomor di kiri, kolom jawaban di kanan.
            # Kita asumsikan sel yang posisinya di > 30% lebar sel adalah jawaban.
            # Atau lebih sederhana: ambil semua sel yang valid, nanti kita sort.
            calon_kotak.append({
                'x': x_tbl + cx,
                'y': y_tbl + cy,
                'w': cw,
                'h': ch,
                'area': cw * ch
            })
            
    if len(calon_kotak) < 40: # Jika kurang dari 40 sel, berarti deteksi gagal
        print(f"[WARNING] Hanya mendeteksi {len(calon_kotak)} sel. Fallback ke Regular Grid.")
        return dapatkan_kotak_regular_grid(img_gray, total_soal)

    # 4. Sorting & Mapping ke Nomor Soal
    # Urutkan berdasarkan Y (Baris), lalu X (Kolom)
    calon_kotak.sort(key=lambda k: (k['y'], k['x']))
    
    # Grup berdasarkan Baris (Y)
    # Karena ada 10 baris, kita bagi range Y menjadi 10 cluster
    y_values = [k['y'] for k in calon_kotak]
    min_y, max_y = min(y_values), max(y_values)
    step_y = (max_y - min_y) / 10
    
    hasil_final = []
    no_soal = 1
    
    # Loop 10 Baris
    for row_idx in range(10):
        y_center = min_y + (row_idx * step_y) + (step_y/2)
        
        # Ambil semua kotak yang Y-nya dekat dengan y_center baris ini
        kotak_di_baris_ini = [k for k in calon_kotak if abs(k['y'] - y_center) < step_y/2]
        
        # Sortir kotak di baris ini berdasarkan X (Kiri ke Kanan)
        kotak_di_baris_ini.sort(key=lambda k: k['x'])
        
        # Dalam satu baris, seharusnya ada 5 kolom jawaban (untuk soal 1-10, 11-20, dst)
        # Tapi struktur tabel Andini: 
        # Kolom 1: No 1-10
        # Kolom 2: No 11-20
        # ...
        # Jadi dalam 1 baris visual, ada 5 grup kotak jawaban.
        
        # Kita asumsikan setiap "grup" memiliki 5 kotak jawaban (A,B,C,D,E)
        # Tapi karena kita mendeteksi SEL, maka setiap sel adalah satu kemungkinan jawaban.
        # Mari kita ambil 5 sel pertama di baris ini sebagai perwakilan kolom utama? 
        # TIDAK. Struktur Andini: Setiap sel adalah SATU JAWABAN.
        # Jadi dalam 1 baris visual, ada 50/10 = 5 SEL? TIDAK.
        # Lihat gambar: Ada 5 KOLOM UTAMA. Setiap kolom utama punya 10 BARIS.
        # Jadi total sel = 50.
        
        # Revisi Logika Sorting:
        # Kita sudah sort semua calon_kotak by Y then X.
        # Maka urutan list `calon_kotak` seharusnya sudah sesuai urutan baca:
        # Baris 1: Sel 1 (Soal 1), Sel 2 (Soal 11), Sel 3 (Soal 21)... -> SALAH!
        # Urutan baca manusia: Soal 1, Soal 2... Soal 10. Lalu baris berikutnya?
        # TIDAK. Di tabel Andini:
        # Baris 1 Visual: Soal 1, Soal 11, Soal 21, Soal 31, Soal 41.
        # Baris 2 Visual: Soal 2, Soal 12, Soal 22, Soal 32, Soal 42.
        
        # Jadi, jika kita sort by Y then X, kita dapat:
        # Index 0: Soal 1
        # Index 1: Soal 11
        # Index 2: Soal 21
        # Index 3: Soal 31
        # Index 4: Soal 41
        # Index 5: Soal 2
        # ...
        
        # Mapping Index ke Nomor Soal:
        # Jika index i, maka:
        # row_visual = i // 5
        # col_visual = i % 5
        # no_soal = (col_visual * 10) + (row_visual + 1)
        
    # Mari kita gunakan logika mapping index langsung dari list yang sudah sorted
    for i, kotak in enumerate(calon_kotak[:50]): # Ambil hanya 50 teratas
        row_visual = i // 5
        col_visual = i % 5
        
        no_soal = (col_visual * 10) + (row_visual + 1)
        
        if no_soal > 50:
            continue
            
        hasil_final.append({
            'no': no_soal,
            'x': kotak['x'],
            'y': kotak['y'],
            'w': kotak['w'],
            'h': kotak['h']
        })
        
    hasil_final.sort(key=lambda item: item['no'])
    return hasil_final


def dapatkan_kotak_regular_grid(img_gray, total_soal=50):
    """Fallback: Pembagian rata jika deteksi tabel gagal."""
    h, w = img_gray.shape
    margin_top = int(h * 0.15)
    margin_bot = int(h * 0.10)
    margin_left = int(w * 0.05)
    margin_right = int(w * 0.05)
    
    area_h = h - margin_top - margin_bot
    area_w = w - margin_left - margin_right
    
    step_y = area_h / 10
    step_x = area_w / 5
    
    hasil = []
    no = 1
    for col in range(5):
        x_start = margin_left + (col * step_x)
        x_kotak = int(x_start + (step_x * 0.2)) # Geser dikit ke kanan dari batas kolom
        
        for row in range(10):
            y_start = margin_top + (row * step_y)
            y_kotak = int(y_start + (step_y * 0.2)) # Geser dikit ke bawah
            
            hasil.append({
                'no': no,
                'x': x_kotak,
                'y': y_kotak,
                'w': int(step_x * 0.6),
                'h': int(step_y * 0.6)
            })
            no += 1
    return hasil

# ──────────────────────────── GET KOTAK VALID ────────────────────────────────
def dapatkan_kotak_valid(img, total_soal=50):
    try:
        img_lurus = luruskan_kertas(img)
        if img_lurus is None:
            return [], None, None

        gray = cv2.cvtColor(img_lurus, cv2.COLOR_BGR2GRAY)

        # Shadow removal
        dilated  = cv2.dilate(gray, np.ones((5, 5), np.uint8))
        bg       = cv2.medianBlur(dilated, 15)
        diff     = 255 - cv2.absdiff(gray, bg)
        norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Gunakan Hybrid Detection
        list_kotak = dapatkan_kotak_hybrid(norm_img, total_soal)
        
        return list_kotak, norm_img, img_lurus

    except Exception as e:
        print(f"[ERROR GRID]: {e}")
        return [], None, None


# ──────────────────────── CROP HURUF (FIXED) ─────────────────────────────────
def dapatkan_crop_huruf_dinamis(thresh_kotak, nomor_soal=0):
    """
    Versi Gentle untuk Tulisan Pensil Tipis.
    Fokus: Menebalkan garis pensil sebelum di-crop agar tidak hilang.
    """
    h_k, w_k = thresh_kotak.shape[:2]
    
    # Margin sangat kecil agar tidak memotong huruf mepet tepi
    margin_kiri = 2
    area_isi    = thresh_kotak[2:h_k - 2, margin_kiri:w_k - 2]

    if area_isi.size == 0:
        return np.zeros((10,10), dtype=np.uint8)

    # --- STRATEGI BARU: PEBALAN GARIS (DILATION) ---
    # Karena tulisan pensil tipis, kita tebalkan dulu biar jadi "hitam pekat"
    kernel_dilate = np.ones((2, 2), np.uint8)
    area_tebal = cv2.dilate(area_isi, kernel_dilate, iterations=1)
    
    # Inversi: Hitam jadi Putih (untuk deteksi kontur latar belakang putih)
    tulisan_inv = cv2.bitwise_not(area_tebal)
    
    contours, _ = cv2.findContours(tulisan_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Jika masih tidak ada kontur, kembalikan area asli yang sudah ditebalkan
        return area_tebal

    # Filter kontur: Ambil yang paling masuk akal sebagai huruf
    contours_valid = [
        cnt for cnt in contours
        if cv2.contourArea(cnt) > 5  # Threshold sangat rendah untuk menangkap titik pensil
        and cv2.boundingRect(cnt)[3] > 5  # Tinggi minimal 5px
    ]
    
    if not contours_valid:
        return area_tebal

    c = max(contours_valid, key=cv2.contourArea)
    
    x_h, y_h, w_h, h_h = cv2.boundingRect(c)
    
    # Padding: Ambil area sekitar huruf
    pad = 5
    x_start = max(0, x_h - pad)
    y_start = max(0, y_h - pad)
    x_end   = min(area_tebal.shape[1], x_h + w_h + pad)
    y_end   = min(area_tebal.shape[0], y_h + h_h + pad)

    crop_dinamis = area_tebal[y_start:y_end, x_start:x_end]

    # Jika hasil crop terlalu kecil, resize up biar Tesseract bisa baca
    if crop_dinamis.shape[0] < 20 or crop_dinamis.shape[1] < 20:
        crop_dinamis = cv2.resize(crop_dinamis, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if nomor_soal > 0 and crop_dinamis.size > 0:
        cv2.imwrite(f"{OS_DEBUG_DIR}/soal_{nomor_soal:02d}.jpg", crop_dinamis)

    return crop_dinamis

# ──────────────────────── OCR + VOTING ──────────────────────────────
VALID_HURUF = {'A', 'B', 'C', 'D', 'E'}
CUSTOM_CFG  = r'-l eng --oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEabcde'


def _ocr_satu_crop(crop_img):
    if crop_img is None or crop_img.size == 0:
        return "", 0.0
    
    # Jika pytesseract tersedia, coba gunakan
    if TESSERACT_AVAILABLE:
        try:
            data = pytesseract.image_to_data(crop_img, config=CUSTOM_CFG,
                                             output_type=pytesseract.Output.DICT)
            huruf_list = []
            conf_list  = []
            for txt, conf in zip(data['text'], data['conf']):
                txt = txt.strip().upper()
                if txt in VALID_HURUF and int(conf) > 0:
                    huruf_list.append(txt)
                    conf_list.append(int(conf) / 100.0)
            if huruf_list:
                idx = conf_list.index(max(conf_list))
                return huruf_list[idx], conf_list[idx]
        except Exception:
            pass
        try:
            raw = pytesseract.image_to_string(crop_img, config=CUSTOM_CFG).strip().upper()
            if raw in VALID_HURUF:
                return raw, 0.5
        except Exception:
            pass
    
    # Fallback: gunakan kamus visual template matching
    return "", 0.0


def tebak_tulisan_tangan(img, total_soal):
    print(f"\n[START OCR] Mulai proses scan untuk {total_soal} soal...")
    hasil = []
    
    try:
        # 1. Dapatkan Kotak
        list_kotak, norm_img, img_lurus = dapatkan_kotak_valid(img, total_soal)
        
        if not list_kotak:
            print("[ERROR CRITICAL] list_kotak KOSONG! Grid gagal dibuat.")
            return [{"huruf": "", "conf": 0.0, "votes": 0} for _ in range(total_soal)]
        
        print(f"[INFO] Berhasil membuat grid dengan {len(list_kotak)} kotak.")

        if norm_img is None:
            print("[ERROR] norm_img adalah None. Cek fungsi luruskan_kertas.")
            return [{"huruf": "", "conf": 0.0, "votes": 0} for _ in range(total_soal)]

        # Simpan gambar normalisasi untuk dicek manual
        cv2.imwrite("debug_norm_full.jpg", norm_img)
        print("[DEBUG] Saved 'debug_norm_full.jpg' (Cek apakah tabel terlihat jelas?)")

        kernel_dilate = np.ones((2, 2), np.uint8)
        hasil = [{"huruf": "", "conf": 0.0, "votes": 0} for _ in range(total_soal)]

        # 2. Loop Setiap Kotak
        processed_count = 0
        
        # Import AI Engine sekali di luar loop agar lebih efisien
        from ai_engine import tebak_huruf_ai
        import config_ai # Untuk tahu AI mana yang aktif

        for item in list_kotak:
            no = item['no']
            if no > total_soal:
                continue
            
            x, y, w, h = item['x'], item['y'], item['w'], item['h']
            
            # Safety check bounds
            if y+h > norm_img.shape[0] or x+w > norm_img.shape[1]:
                print(f"[SKIP] No.{no} out of bounds")
                continue

            # Ambil ROI (Region of Interest) dari gambar grayscale asli (norm_img)
            roi_gray = norm_img[y:y+h, x:x+w]
            
            if roi_gray.size == 0:
                continue

            # --- PREPROCESSING KHUSUS PENSIL TIPIS ---
            # 1. Thresholding Otsu Lokal
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 2. Dilation: Tebalkan garis pensil
            roi_dilated = cv2.dilate(roi_thresh, kernel_dilate, iterations=1)
            
            # 3. Crop Dinamis
            crop_final = dapatkan_crop_huruf_dinamis(roi_dilated, nomor_soal=no)
            
            if crop_final is None or crop_final.size == 0:
                print(f"[WARN] No.{no} crop_empty setelah preprocessing")
                continue

            # 4. Resize Up: AI Vision butuh resolusi cukup baik
            target_height = 60
            scale = target_height / float(crop_final.shape[0])
            if scale > 0:
                new_width = int(crop_final.shape[1] * scale)
                crop_big = cv2.resize(crop_final, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
            else:
                crop_big = crop_final

            # Simpan hasil preprocessing FINAL untuk dicek (Opsional, bisa dimatikan jika lambat)
            # cv2.imwrite(f"{OS_DEBUG_DIR}/FINAL_No{no:02d}.jpg", crop_big)

            # --- INTEGRASI AI ENGINE (GEMINI / OPENAI) ---
            res_huruf, res_conf = tebak_huruf_ai(crop_big)
            
            if res_huruf and res_huruf in ['A', 'B', 'C', 'D', 'E']:
                hasil[no-1] = {"huruf": res_huruf, "conf": res_conf, "votes": 1}
                processed_count += 1
                
                # Print log singkat biar tidak spam, tapi tetap informatif
                print(f"[AI-{config_ai.ACTIVE_AI}] No.{no:2d} -> '{res_huruf}' (Conf: {res_conf:.2f})")
            else:
                print(f"[WARN] AI Gagal mendeteksi No.{no}. Hasil kosong.")
                # Opsional: Bisa tambahkan fallback ke Tesseract di sini jika mau hybrid murni
                # hasil[no-1] = {"huruf": "", "conf": 0.0, "votes": 0}

            # Progress bar sederhana
            if no % 10 == 0:
                print(f"[PROGRESS] Memproses soal no {no}...")

        print(f"[DONE] Total soal diproses: {processed_count}/{total_soal}")
        print(f"[CHECK] Buka folder '{OS_DEBUG_DIR}' jika perlu cek visual.")

    except Exception as e:
        import traceback
        print(f"[CRASH ERROR] {e}")
        traceback.print_exc()
        hasil = [{"huruf": "", "conf": 0.0, "votes": 0} for _ in range(total_soal)]

    return hasil

# ─────────────────────────────── FRONTEND & ROUTES (SAMA SEPERTI SEBELUMNYA) ────────────────────────────────────
# ... (Salin bagian HTML_TEMPLATE dan Routes dari kode sebelumnya, tidak ada perubahan di sini) ...

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Scan Ujian v2.0 - Presisi Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js"></script>
    <style>
        :root {
            --cyan:      #00f0ff;
            --magenta:   #ff00aa;
            --purple:    #9b00ff;
            --green:     #00ff88;
            --amber:     #ffb800;
            --red:       #ff4455;
            --bg-void:   #040008;
            --bg-card:   #08001a;
            --bg-input:  #06000f;
            --border-dim:#1a0035;
            --text:      #ccd8f0;
            --label:     #6677aa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Exo 2', sans-serif;
            padding: 20px 15px;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--text);
            background-color: var(--bg-void);
            background-image:
                linear-gradient(rgba(0,240,255,0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,240,255,0.025) 1px, transparent 1px);
            background-size: 44px 44px;
        }
        .container {
            width: 100%;
            max-width: 480px;
            background: var(--bg-card);
            padding: 32px 24px 28px;
            border-radius: 3px;
            position: relative;
            border: 1px solid var(--cyan);
            box-shadow:
                0 0 0 1px rgba(0,240,255,0.08),
                0 0 30px rgba(0,240,255,0.12),
                0 0 80px rgba(0,240,255,0.04),
                inset 0 0 60px rgba(0,0,40,0.6);
        }
        .container::before {
            content: '';
            position: absolute;
            top: 6px; left: 6px;
            width: 20px; height: 20px;
            border-top: 2px solid var(--cyan);
            border-left: 2px solid var(--cyan);
            box-shadow: -4px -4px 8px rgba(0,240,255,0.3);
        }
        .container::after {
            content: '';
            position: absolute;
            bottom: 6px; right: 6px;
            width: 20px; height: 20px;
            border-bottom: 2px solid var(--cyan);
            border-right: 2px solid var(--cyan);
            box-shadow: 4px 4px 8px rgba(0,240,255,0.3);
        }
        .container > h2::before {
            content: '';
            display: block;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--cyan), var(--purple), var(--magenta), transparent);
            margin-bottom: 22px;
            margin-top: -4px;
            border-radius: 2px;
            box-shadow: 0 0 12px rgba(0,240,255,0.5);
        }
        h2 {
            font-family: 'Orbitron', sans-serif;
            color: var(--cyan);
            font-weight: 900;
            font-size: 17px;
            text-align: center;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            text-shadow: 0 0 10px rgba(0,240,255,0.8), 0 0 30px rgba(0,240,255,0.3);
            margin-bottom: 26px;
            position: relative;
        }
        h2 span { position: relative; display: inline-block; }
        h2 span::before, h2 span::after {
            content: attr(data-text);
            position: absolute;
            top: 0; left: 0;
            width: 100%;
        }
        h2 span::before {
            color: var(--magenta);
            animation: glitch-top 5s infinite;
            clip-path: inset(0 0 60% 0);
        }
        h2 span::after {
            color: var(--cyan);
            animation: glitch-bot 5s infinite;
            clip-path: inset(60% 0 0 0);
        }
        @keyframes glitch-top {
            0%, 88%, 100% { opacity:0; transform:translate(0,0); }
            90% { opacity:1; transform:translate(-3px, 0); }
            92% { opacity:1; transform:translate(3px, 0); }
            94% { opacity:1; transform:translate(-1px, 0); }
            96% { opacity:0; }
        }
        @keyframes glitch-bot {
            0%, 88%, 100% { opacity:0; transform:translate(0,0); }
            91% { opacity:1; transform:translate(3px, 0); }
            93% { opacity:1; transform:translate(-2px, 0); }
            95% { opacity:1; transform:translate(1px, 0); }
            96% { opacity:0; }
        }
        label {
            display: block;
            text-align: left;
            font-family: 'Share Tech Mono', monospace;
            font-size: 11px;
            font-weight: 400;
            margin-bottom: 7px;
            color: var(--label);
            text-transform: uppercase;
            letter-spacing: 0.13em;
        }
        .form-group { margin-bottom: 18px; }
        input, select {
            width: 100%;
            padding: 12px 14px;
            border-radius: 2px;
            border: 1px solid var(--border-dim);
            border-bottom: 2px solid rgba(0,240,255,0.4);
            font-size: 14px;
            color: var(--text);
            background-color: var(--bg-input);
            outline: none;
            transition: all 0.2s;
            font-family: 'Exo 2', sans-serif;
            font-weight: 500;
        }
        input::placeholder { color: #2a3050; }
        input:focus, select:focus {
            border-color: var(--cyan);
            border-bottom-color: var(--cyan);
            background-color: rgba(0,240,255,0.03);
            box-shadow: 0 0 0 1px rgba(0,240,255,0.15), 0 0 20px rgba(0,240,255,0.1);
            color: #fff;
        }
        select option { background: #08001a; color: var(--text); }
        select optgroup { color: var(--cyan); font-weight: 700; background: #06000f; }
        .video-container {
            position: relative;
            width: 100%;
            border-radius: 2px;
            overflow: hidden;
            background-color: #000;
            margin: 20px 0;
            aspect-ratio: 4/3;
            border: 1px solid rgba(0,240,255,0.25);
            box-shadow: 0 0 20px rgba(0,240,255,0.08), inset 0 0 30px rgba(0,0,20,0.8);
        }
        .video-container::after {
            content: '';
            position: absolute;
            inset: 0;
            background: repeating-linear-gradient(
                to bottom, transparent 0px, transparent 2px,
                rgba(0,0,0,0.14) 2px, rgba(0,0,0,0.14) 4px
            );
            pointer-events: none;
            z-index: 4;
        }
        .video-container::before {
            content: '';
            position: absolute;
            inset: 10px;
            border: 1px solid rgba(0,240,255,0.18);
            border-radius: 1px;
            pointer-events: none;
            z-index: 5;
            box-shadow: inset 0 0 20px rgba(0,240,255,0.04);
        }
        #video { width: 100%; height: 100%; object-fit: cover; }
        .preview-wrapper {
            max-width: 100%;
            max-height: 400px;
            display: none;
            margin: 20px 0;
            border-radius: 2px;
            overflow: hidden;
            border: 1px solid var(--cyan);
            box-shadow: 0 0 20px rgba(0,240,255,0.2);
        }
        #photo-preview { max-width: 100%; display: block; }
        .flash-btn {
            position: absolute;
            top: 12px; right: 12px;
            width: 42px; height: 42px;
            border-radius: 2px;
            background: rgba(0,0,20,0.85);
            border: 1px solid rgba(0,240,255,0.35);
            color: var(--cyan);
            font-size: 18px;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10;
            backdrop-filter: blur(6px);
            transition: all 0.2s;
        }
        .flash-btn:hover { border-color: var(--amber); color: var(--amber); box-shadow: 0 0 14px rgba(255,184,0,0.4); }
        .flash-active { background: rgba(255,184,0,0.12) !important; color: var(--amber) !important; border-color: var(--amber) !important; box-shadow: 0 0 20px var(--amber) !important; }
        .action-container { display: flex; flex-direction: column; gap: 10px; }
        button {
            width: 100%;
            font-weight: 700;
            font-size: 12px;
            padding: 15px 16px;
            border: none;
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.18s;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-family: 'Share Tech Mono', monospace;
            position: relative;
            overflow: hidden;
        }
        button::after {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 60%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
            transition: left 0.35s;
        }
        button:not(:disabled):hover::after { left: 150%; }
        button:not(:disabled):active { transform: translateY(1px); }
        .btn-blue { background: transparent; border: 1px solid var(--cyan); color: var(--cyan); text-shadow: 0 0 8px rgba(0,240,255,0.6); box-shadow: 0 0 12px rgba(0,240,255,0.15); }
        .btn-blue:not(:disabled):hover { background: rgba(0,240,255,0.07); box-shadow: 0 0 22px rgba(0,240,255,0.35); }
        .btn-orange { background: transparent; border: 1px solid var(--magenta); color: var(--magenta); text-shadow: 0 0 8px rgba(255,0,170,0.6); box-shadow: 0 0 12px rgba(255,0,170,0.15); }
        .btn-orange:not(:disabled):hover { background: rgba(255,0,170,0.07); box-shadow: 0 0 22px rgba(255,0,170,0.35); }
        .btn-green-crop { background: transparent; border: 1px solid var(--green); color: var(--green); text-shadow: 0 0 8px rgba(0,255,136,0.6); box-shadow: 0 0 12px rgba(0,255,136,0.15); }
        .btn-green-crop:not(:disabled):hover { background: rgba(0,255,136,0.07); box-shadow: 0 0 22px rgba(0,255,136,0.35); }
        .btn-gray { background: transparent; border: 1px solid #2a3a50; color: #5a6a80; }
        .btn-gray:not(:disabled):hover { border-color: #3a5a70; color: #8899bb; }
        .btn-green { background: transparent; border: 1px solid var(--green); color: var(--green); text-shadow: 0 0 8px rgba(0,255,136,0.6); box-shadow: 0 0 12px rgba(0,255,136,0.15); margin-top: 15px; }
        .btn-green:not(:disabled):hover { background: rgba(0,255,136,0.07); box-shadow: 0 0 22px rgba(0,255,136,0.35); }
        button:disabled { background: transparent !important; border: 1px solid #1a2030 !important; color: #2a3545 !important; text-shadow: none !important; box-shadow: none !important; cursor: not-allowed; transform: none !important; }
        button:disabled::after { display: none; }

        /* ── Review Box ─ */
        .review-box {
            margin-top: 24px;
            padding: 20px;
            border-radius: 2px;
            background-color: rgba(0,240,255,0.02);
            display: none;
            border: 1px solid var(--border-dim);
            border-top: 2px solid var(--cyan);
            box-shadow: 0 0 20px rgba(0,240,255,0.05);
        }
        .review-box > label { color: var(--cyan); font-size: 11px; letter-spacing: 0.12em; text-shadow: 0 0 8px rgba(0,240,255,0.5); margin-bottom: 12px; }
        .list-jawaban {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin: 14px 0;
            max-height: 320px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .list-jawaban::-webkit-scrollbar { width: 3px; }
        .list-jawaban::-webkit-scrollbar-track { background: var(--bg-void); }
        .list-jawaban::-webkit-scrollbar-thumb { background: var(--cyan); }

        /* ── Item Soal + Confidence Badge ── */
        .item-soal {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-input);
            padding: 7px 10px;
            border-radius: 2px;
            border: 1px solid var(--border-dim);
            border-left: 2px solid var(--purple);
            font-family: 'Share Tech Mono', monospace;
            font-size: 12px;
            color: var(--label);
            position: relative;
        }
        /* Soal yang ragu-ragu (votes < 2) → border merah + glow */
        .item-soal.ragu {
            border-left-color: var(--red);
            background: rgba(255,68,85,0.06);
            box-shadow: inset 0 0 12px rgba(255,68,85,0.08);
        }
        .item-soal.ragu::after {
            content: '?';
            position: absolute;
            top: 3px; right: 36px;
            font-size: 9px;
            color: var(--red);
            font-weight: 900;
            opacity: 0.8;
        }
        .item-soal input {
            width: 44px;
            padding: 5px 4px;
            text-align: center;
            border: 1px solid var(--border-dim);
            border-radius: 2px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 13px;
            font-weight: 700;
            color: var(--cyan);
            background: var(--bg-void);
        }
        .item-soal input:focus { border-color: var(--cyan); outline: none; box-shadow: 0 0 8px rgba(0,240,255,0.3); }
        .item-soal.ragu input { color: var(--red); border-color: rgba(255,68,85,0.4); }

        /* Confidence bar kecil di bawah item */
        .conf-bar {
            position: absolute;
            bottom: 0; left: 0;
            height: 2px;
            border-radius: 0 0 0 2px;
            transition: width 0.3s;
        }

        /* ── Stat row ── */
        .stat-row {
            display: flex;
            justify-content: space-between;
            font-family: 'Share Tech Mono', monospace;
            font-size: 10px;
            color: var(--label);
            margin-bottom: 10px;
            letter-spacing: 0.08em;
        }
        .stat-row span { color: var(--cyan); }

        /* ── Result box ── */
        .result-box {
            display: none;
            margin-top: 20px;
            padding: 20px;
            border-radius: 2px;
            background: rgba(0,255,136,0.03);
            text-align: left;
            border: 1px solid var(--green);
            box-shadow: 0 0 30px rgba(0,255,136,0.12), inset 0 0 30px rgba(0,255,136,0.03);
        }
        #res-skor {
            font-family: 'Orbitron', sans-serif;
            font-size: 38px;
            font-weight: 900;
            color: var(--green);
            text-align: center;
            margin-top: 10px;
            text-shadow: 0 0 15px rgba(0,255,136,0.7), 0 0 45px rgba(0,255,136,0.3);
            letter-spacing: 0.06em;
            animation: pulse-score 2s ease-in-out infinite;
        }
        @keyframes pulse-score {
            0%, 100% { text-shadow: 0 0 15px rgba(0,255,136,0.7), 0 0 45px rgba(0,255,136,0.3); }
            50%       { text-shadow: 0 0 25px rgba(0,255,136,1), 0 0 70px rgba(0,255,136,0.5); }
        }
        .version-tag {
            font-family: 'Share Tech Mono', monospace;
            font-size: 10px;
            color: rgba(0,240,255,0.3);
            text-align: center;
            letter-spacing: 0.15em;
            margin-bottom: 20px;
            margin-top: -12px;
        }
    </style>
</head>
<body>
<div class="container">
    <h2><span data-text="Scanner Lembar Ujian">Scanner Lembar Ujian</span></h2>
    <p class="version-tag">// SYS:SCAN v2.0 · CONTENT-BASED ANCHOR · CONF ENGINE</p>

    <div class="form-group">
        <label for="kelas">// Pilih Tingkat &amp; Kelas</label>
        <select id="kelas">
            <optgroup label="SMP">
                <option value="SMP_7_A">Kelas 7 A (SMP)</option>
                <option value="SMP_7_B">Kelas 7 B (SMP)</option>
                <option value="SMP_7_C">Kelas 7 C (SMP)</option>
                <option value="SMP_7_D">Kelas 7 D (SMP)</option>
                <option value="SMP_7_E">Kelas 7 E (SMP)</option>
                <option value="SMP_7_F">Kelas 7 F (SMP)</option>
                <option value="SMP_8_A">Kelas 8 A (SMP)</option>
                <option value="SMP_8_B">Kelas 8 B (SMP)</option>
                <option value="SMP_8_C">Kelas 8 C (SMP)</option>
                <option value="SMP_8_D">Kelas 8 D (SMP)</option>
                <option value="SMP_8_E">Kelas 8 E (SMP)</option>
                <option value="SMP_8_F">Kelas 8 F (SMP)</option>
                <option value="SMP_9_A">Kelas 9 A (SMP)</option>
                <option value="SMP_9_B">Kelas 9 B (SMP)</option>
                <option value="SMP_9_C">Kelas 9 C (SMP)</option>
                <option value="SMP_9_D">Kelas 9 D (SMP)</option>
                <option value="SMP_9_E">Kelas 9 E (SMP)</option>
                <option value="SMP_9_F">Kelas 9 F (SMP)</option>
            </optgroup>
            <optgroup label="SMA">
                <option value="SMA_10_A">Kelas 10 A (SMA)</option>
                <option value="SMA_10_B">Kelas 10 B (SMA)</option>
                <option value="SMA_10_C">Kelas 10 C (SMA)</option>
                <option value="SMA_10_D">Kelas 10 D (SMA)</option>
                <option value="SMA_10_E">Kelas 10 E (SMA)</option>
                <option value="SMA_10_F">Kelas 10 F (SMA)</option>
                <option value="SMA_11_A">Kelas 11 A (SMA)</option>
                <option value="SMA_11_B">Kelas 11 B (SMA)</option>
                <option value="SMA_11_C">Kelas 11 C (SMA)</option>
                <option value="SMA_11_D">Kelas 11 D (SMA)</option>
                <option value="SMA_11_E">Kelas 11 E (SMA)</option>
                <option value="SMA_11_F">Kelas 11 F (SMA)</option>
                <option value="SMA_12_A">Kelas 12 A (SMA)</option>
                <option value="SMA_12_B">Kelas 12 B (SMA)</option>
                <option value="SMA_12_C">Kelas 12 C (SMA)</option>
                <option value="SMA_12_D">Kelas 12 D (SMA)</option>
                <option value="SMA_12_E">Kelas 12 E (SMA)</option>
                <option value="SMA_12_F">Kelas 12 F (SMA)</option>
            </optgroup>
        </select>
    </div>

    <div class="form-group">
        <label for="nama_siswa">// Nama Lengkap Siswa</label>
        <input type="text" id="nama_siswa" placeholder="Ketik nama lengkap..." required>
    </div>

    <div id="cam-area" class="video-container">
        <button id="btn-flash" class="flash-btn">⚡</button>
        <video id="video" autoplay playsinline></video>
    </div>

    <div id="preview-area" class="preview-wrapper">
        <img id="photo-preview" alt="Preview Foto Kertas">
    </div>

    <div class="action-container">
        <button id="btn-capture"     class="btn-blue">📸 LANGKAH 1: AMBIL FOTO KERTAS</button>
        <button id="btn-crop-confirm" class="btn-green-crop" style="display:none;">✂️ LANGKAH 1.5: KUNCI AREA POTONGAN</button>
        <button id="btn-scan"        class="btn-orange" style="display:none;">🎯 LANGKAH 2: PROSES SCAN JAWABAN</button>
        <button id="btn-retake"      class="btn-gray"   style="display:none;">🔄 FOTO ULANG</button>
    </div>

    <div id="review-box" class="review-box">
        <label>✏️ KOREKSI HASIL TEBAKAN OCR:</label>
        <!-- Stat row: jumlah ragu -->
        <div class="stat-row" id="stat-row" style="display:none;">
            RAGU-RAGU: <span id="stat-ragu">0</span> soal &nbsp;|&nbsp; YAKIN: <span id="stat-yakin">0</span> soal
        </div>
        <div id="list-jawaban" class="list-jawaban"></div>
        <button id="btn-konfirmasi" class="btn-green">LANGKAH 3: SIMPAN &amp; HITUNG NILAI</button>
    </div>

    <div id="result-box" class="result-box">
        <h1 id="res-skor"></h1>
    </div>
</div>

<script>
    const video        = document.getElementById('video');
    const camArea      = document.getElementById('cam-area');
    const previewArea  = document.getElementById('preview-area');
    const photoPreview = document.getElementById('photo-preview');
    const btnCapture   = document.getElementById('btn-capture');
    const btnCropConfirm = document.getElementById('btn-crop-confirm');
    const btnScan      = document.getElementById('btn-scan');
    const btnRetake    = document.getElementById('btn-retake');
    const btnFlash     = document.getElementById('btn-flash');
    const reviewBox    = document.getElementById('review-box');
    const listJawaban  = document.getElementById('list-jawaban');
    const btnKonfirmasi = document.getElementById('btn-konfirmasi');
    const resultBox    = document.getElementById('result-box');
    const statRow      = document.getElementById('stat-row');
    const statRagu     = document.getElementById('stat-ragu');
    const statYakin    = document.getElementById('stat-yakin');

    let videoTrack = null;
    let flashStatus = false;
    let cropper = null;
    let currentBlob = null;

    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
        .then(stream => {
            video.srcObject = stream;
            videoTrack = stream.getVideoTracks()[0];
        }).catch(() => alert("Akses kamera ditolak!"));

    btnFlash.onclick = async () => {
        if (!videoTrack) return;
        try {
            flashStatus = !flashStatus;
            await videoTrack.applyConstraints({ advanced: [{ torch: flashStatus }] });
            btnFlash.classList.toggle('flash-active', flashStatus);
        } catch { alert("Senter tidak didukung perangkat ini."); }
    };

    btnCapture.onclick = async () => {
        const nama = document.getElementById('nama_siswa').value.trim();
        if (!nama) { alert("Isi nama siswanya dulu, bos!"); return; }

        const canvas = document.createElement('canvas');
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        photoPreview.src = canvas.toDataURL('image/jpeg');

        if (videoTrack && flashStatus) {
            try {
                flashStatus = false;
                await videoTrack.applyConstraints({ advanced: [{ torch: false }] });
                btnFlash.classList.remove('flash-active');
            } catch {}
        }

        camArea.style.display    = "none";
        previewArea.style.display = "block";
        if (cropper) cropper.destroy();

        cropper = new Cropper(photoPreview, {
            viewMode: 1, dragMode: 'move', autoCropArea: 0.9,
            restore: false, guides: false, center: false,
            highlight: false, cropBoxMovable: true, cropBoxResizable: true,
            toggleDragModeOnDblclick: false
        });

        btnCapture.style.display     = "none";
        btnCropConfirm.style.display = "block";
        btnRetake.style.display      = "block";
    };

    btnCropConfirm.onclick = () => {
        if (!cropper) return;
        cropper.getCroppedCanvas({ maxWidth: 1200, maxHeight: 1200 }).toBlob(blob => {
            currentBlob = blob;
            const url = URL.createObjectURL(blob);
            cropper.destroy();
            cropper = null;
            photoPreview.src = url;
            btnCropConfirm.style.display = "none";
            btnScan.style.display        = "block";
        }, 'image/jpeg');
    };

    btnRetake.onclick = () => {
        if (cropper) { cropper.destroy(); cropper = null; }
        camArea.style.display        = "block";
        previewArea.style.display    = "none";
        btnCapture.style.display     = "block";
        btnCropConfirm.style.display = "none";
        btnScan.style.display        = "none";
        btnRetake.style.display      = "none";
        reviewBox.style.display      = "none";
        resultBox.style.display      = "none";
    };

    btnScan.onclick = async () => {
        const kelas = document.getElementById('kelas').value;
        btnScan.innerText = "MENGANALISIS DATA LJK...";
        btnScan.disabled  = true;

        const formData = new FormData();
        formData.append('gambar_ujian', currentBlob, 'scan.jpg');
        formData.append('kelas', kelas);

        try {
            const response = await fetch('/proses_scan', { method: 'POST', body: formData });
            const result   = await response.json();

            if (result.status === "success") {
                listJawaban.innerHTML = "";
                const totalSoal  = result.total_soal;
                const jawabanRaw = result.jawaban_ocr.slice(0, totalSoal);

                let jumlahRagu  = 0;
                let jumlahYakin = 0;

                jawabanRaw.forEach((item, index) => {
                    const nomorSoal = index + 1;
                    const huruf  = (item.huruf || "").toUpperCase();
                    const votes  = item.votes  || 0;
                    const conf   = item.conf   || 0;

                    const isRagu = huruf === "" || votes < 2 || conf < 0.35;
                    if (isRagu) jumlahRagu++;
                    else        jumlahYakin++;

                    const div = document.createElement('div');
                    div.className = 'item-soal' + (isRagu ? ' ragu' : '');

                    const barColor = conf > 0.6 ? '#00ff88' : conf > 0.35 ? '#ffb800' : '#ff4455';
                    const barWidth = Math.round(conf * 100);

                    div.innerHTML = `
                        <span>No.${nomorSoal}</span>
                        <input type="text" maxlength="1"
                               class="input-jawaban-siswa"
                               data-no="${nomorSoal}"
                               value="${huruf}"
                               oninput="this.value=this.value.toUpperCase();
                                        this.closest('.item-soal').classList.remove('ragu');">
                        <div class="conf-bar"
                             style="width:${barWidth}%;background:${barColor};"></div>
                    `;
                    listJawaban.appendChild(div);
                });

                statRagu.innerText  = jumlahRagu;
                statYakin.innerText = jumlahYakin;
                statRow.style.display = "flex";

                reviewBox.style.display = "block";

                const soalRaguPertama = listJawaban.querySelector('.ragu');
                if (soalRaguPertama) {
                    setTimeout(() => soalRaguPertama.scrollIntoView({ behavior: 'smooth', block: 'center' }), 300);
                }

            } else {
                alert("Gagal: " + result.message);
            }
        } catch {
            alert("Gagal menyambung ke backend Python!");
        } finally {
            btnScan.innerText = " LANGKAH 2: PROSES SCAN JAWABAN";
            btnScan.disabled  = false;
        }
    };

    btnKonfirmasi.onclick = async () => {
        const nama   = document.getElementById('nama_siswa').value.trim();
        const kelas  = document.getElementById('kelas').value;
        const inputs = document.querySelectorAll('.input-jawaban-siswa');

        let dataJawabanFinal = {};
        inputs.forEach(inp => {
            dataJawabanFinal[inp.getAttribute('data-no')] = inp.value.trim().toUpperCase();
        });

        btnKonfirmasi.disabled = true;
        const formData = new FormData();
        formData.append('gambar_ujian', currentBlob, 'scan.jpg');
        formData.append('nama',          nama);
        formData.append('kelas',         kelas);
        formData.append('jawaban_final', JSON.stringify(dataJawabanFinal));

        try {
            const response = await fetch('/konfirmasi_nilai', { method: 'POST', body: formData });
            const result   = await response.json();
            if (result.status === "success") {
                document.getElementById('res-skor').innerText = "SKOR: " + result.skor;
                resultBox.style.display = "block";
                reviewBox.style.display = "none";
            }
        } catch {
            alert("Gagal menyimpan data!");
        } finally {
            btnKonfirmasi.disabled = false;
        }
    };
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return Response(HTML_TEMPLATE, mimetype='text/html')


@app.route('/proses_scan', methods=['POST'])
def proses_scan():
    try:
        kelas     = request.form.get('kelas', 'KOSONG')
        file_bytes = request.files['gambar_ujian'].read()
        np_img    = np.frombuffer(file_bytes, np.uint8)
        img       = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        h_check, w_check = img.shape[:2]
        if w_check < 150 or h_check < 150:
            return jsonify({"status": "error",
                            "message": "Resolusi terlalu kecil. Deketin kamera ke kertas!"}), 400

        # Hapus rotasi otomatis karena user sudah crop manual
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        path_kunci   = f"kunci_jawaban/{kelas}.txt"
        kunci_jawaban = load_kunci_jawaban(path_kunci)
        total_soal   = len(kunci_jawaban) if kunci_jawaban else 50 # Default 50 jika kunci kosong

        jawaban_ocr = tebak_tulisan_tangan(img, total_soal)

        # Jangan return error jika hasil kosong, biarkan frontend menangani
        # if jawaban_ocr == ["FAILED_DESKEW"]:
        #     return jsonify({...})

        return jsonify({
            "status":     "success",
            "jawaban_ocr": jawaban_ocr,
            "total_soal": total_soal
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/konfirmasi_nilai', methods=['POST'])
def konfirmasi_nilai():
    try:
        nama          = request.form.get('nama', 'Tanpa_Nama').strip()
        kelas         = request.form.get('kelas', 'KOSONG')
        jawaban_final = json.loads(request.form.get('jawaban_final', '{}'))

        file_bytes = request.files['gambar_ujian'].read()
        np_img     = np.frombuffer(file_bytes, np.uint8)
        img        = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        # img        = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        path_kunci    = f"kunci_jawaban/{kelas}.txt"
        kunci_jawaban = load_kunci_jawaban(path_kunci)
        total_soal    = len(kunci_jawaban) if kunci_jawaban else 50

        nilai_benar = sum(
            1 for no, kunci in kunci_jawaban.items()
            if jawaban_final.get(str(no), '').strip().upper() == kunci
        )
        skor_akhir = int((nilai_benar / total_soal) * 100) if total_soal > 0 else 0

        os.makedirs(f"arsip_scan/{kelas}", exist_ok=True)
        os.makedirs('hasil_nilai',  exist_ok=True)
        os.makedirs('kamus_huruf',  exist_ok=True)

        tgl = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(
            f"arsip_scan/{kelas}/{tgl}_{nama.replace(' ', '_')}.jpg",
            img, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        csv_path    = f"hasil_nilai/nilai_{kelas}.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a') as f:
            if not file_exists:
                f.write("Tanggal,Nama,Kelas,Skor\n")
            f.write(f"{tgl},{nama},{kelas},{skor_akhir}\n")

        # Panen dataset
        list_kotak, norm_img, _ = dapatkan_kotak_valid(img, total_soal)
        if list_kotak and norm_img is not None:
            _, thresh_normal = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for no, huruf_pilihan in jawaban_final.items():
                try:
                    no_int = int(no)
                    if (no_int in kunci_jawaban
                            and any(item['no'] == no_int for item in list_kotak)
                            and huruf_pilihan in ['A', 'B', 'C', 'D', 'E']):
                        item_kotak = next(item for item in list_kotak if item['no'] == no_int)
                        x, y, w, h = item_kotak['x'], item_kotak['y'], item_kotak['w'], item_kotak['h']
                        kotak_normal = thresh_normal[y:y + h, x:x + w]
                        crop_huruf   = dapatkan_crop_huruf_dinamis(kotak_normal, nomor_soal=0)
                        if crop_huruf is not None and crop_huruf.size > 0:
                            path_panen = f"kamus_huruf/{huruf_pilihan}_{tgl}_no{no}.jpg"
                            cv2.imwrite(path_panen, crop_huruf, [cv2.IMWRITE_JPEG_QUALITY, 80])
                except (ValueError, StopIteration):
                    continue

        return jsonify({"status": "success", "nama": nama, "kelas": kelas, "skor": skor_akhir})

    except Exception as e:
        print(f"[ERROR KONFIRMASI]: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
# telegram_bot.py
import os
import tempfile
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from ai_engine import get_ai_engine

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# PATH DINAMIS (Agar tidak crash di environment selain PC Anda)
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Halo! Saya Bot Scanner LJK Nikto.\n\n"
        "Kirimkan foto LJK yang jelas, dan saya akan membacanya menggunakan AI.\n"
        "Perintah:\n"
        "/scan - Mulai scan LJK\n"
        "/help - Bantuan penggunaan"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Cara Penggunaan:\n"
        "1. Kirim foto LJK dalam format JPG/PNG\n"
        "2. Pastikan pencahayaan cukup terang\n"
        "3. Tunggu beberapa detik hingga hasil muncul\n\n"
        "💡 Tips: Gunakan mode dokumen saat mengirim foto agar kualitas tidak dikompres."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Sedang mengunduh foto...")
    temp_path = None
    
    try:
        # Ambil file resolusi tertinggi
        photo_file = await update.message.photo[-1].get_file() if update.message.photo else await update.message.document.get_file()
        
        fd, temp_path = tempfile.mkstemp(suffix='.jpg', dir=UPLOAD_DIR)
        os.close(fd)
        
        await photo_file.download_to_drive(temp_path)
        await msg.edit_text("🔍 Sedang memproses LJK dengan AI...")
        
        engine = get_ai_engine()
        result = engine.scan_ljk(temp_path)
        
        if 'error' in result and len(result) == 1:
            await msg.edit_text(f"❌ Gagal memindai:\n{result['error']}")
        else:
            jawaban = result.get('jawaban_siswa', [])
            kualitas = result.get('kualitas_gambar', 'N/A')
            
            response_text = f"✅ Hasil Scan LJK\n"
            response_text += f"📷 Kualitas Gambar: {kualitas}\n"
            response_text += f"📝 Total Soal: {len(jawaban)}\n\n"
            response_text += "Jawaban Siswa:\n"
            
            for i, ans in enumerate(jawaban[:20], 1):
                response_text += f"{i}. {ans}\n"
                
            if len(jawaban) > 20:
                response_text += f"...dan {len(jawaban)-20} soal lainnya."
                
            await msg.edit_text(response_text)
            
    except Exception as e:
        await msg.edit_text(f"⚠️ Terjadi kesalahan sistem:\n{str(e)}")
        
    finally:
        # PEMBERSIHAN OTOMATIS: Dijalankan walau prosesnya error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN tidak diset. Bot dibatalkan.")
        return
        
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    
    # Tangkap file berupa gambar kompresi dan berkas asli
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_photo))
    
    print("🤖 Bot Telegram berjalan...")
    app.run_polling()

if __name__ == "__main__":
    main()
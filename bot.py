import os, cv2, numpy as np, time
from datetime import datetime
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters
)
from tensorflow.keras.models import load_model
from fpdf import FPDF

# ================= CONFIG =================
TOKEN = "8118091817:AAH7XHU7x5Ft6JnqPWA3kj-O2ZFE7sB92wY"

REFERENCE, TEST = range(2)

REF_DIR = "data/refs"
REPORT_DIR = "reports"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load trained CNN
model = load_model("model.h5")

# ================= IMAGE PREPROCESS =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (300, 150))
    img = img.astype("float32") / 255.0
    return img.reshape(1, 150, 300, 1)

# ================= BEST-K AGGREGATION =================
def best_k(scores, k=3):
    scores = sorted(scores, reverse=True)
    return float(np.mean(scores[:min(k, len(scores))]))

# ================= PDF =================
def generate_pdf(score, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "CNN Signature Verification Report", ln=True)
    pdf.ln(5)
    pdf.multi_cell(
        0, 8,
        f"Confidence Score: {score:.2f}%\n"
        f"Result: {result}\n\n"
        "Model: Siamese Convolutional Neural Network"
    )

    path = f"{REPORT_DIR}/report_{int(datetime.now().timestamp())}.pdf"
    pdf.output(path)
    return path

# ================= BOT HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "✍️ CNN Signature Verification Bot\n\n"
        "Upload reference signatures (2–5 images).\n"
        "Then type /verify"
    )
    return REFERENCE

async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    user_dir = os.path.join(REF_DIR, uid)
    os.makedirs(user_dir, exist_ok=True)

    photo = await update.message.photo[-1].get_file()
    path = os.path.join(user_dir, f"ref_{len(os.listdir(user_dir))+1}.jpg")
    await photo.download_to_drive(path)

    await update.message.reply_text("✅ Reference saved. Upload more or type /verify")
    return REFERENCE

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Upload ONE test signature")
    return TEST

async def test_signature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    user_dir = os.path.join(REF_DIR, uid)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    test_img = preprocess(test_path)

    scores = []
    for ref in os.listdir(user_dir):
        ref_img = preprocess(os.path.join(user_dir, ref))
        pred = model.predict([ref_img, test_img], verbose=0)
        scores.append(pred[0][0] * 100)

    score = best_k(scores)
    result = "MATCH ✅" if score >= 75 else "MISMATCH ❌"

    pdf_path = generate_pdf(score, result)

    await update.message.reply_text(
        f"🔍 Result\n\n"
        f"Confidence: {score:.2f}%\n"
        f"Result: {result}"
    )

    await update.message.reply_document(open(pdf_path, "rb"))

    # Cleanup
    time.sleep(0.5)
    for f in os.listdir(user_dir):
        os.remove(os.path.join(user_dir, f))
    os.rmdir(user_dir)

    return ConversationHandler.END

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REFERENCE: [
                MessageHandler(filters.PHOTO, save_reference),
                CommandHandler("verify", verify)
            ],
            TEST: [MessageHandler(filters.PHOTO, test_signature)]
        },
        fallbacks=[CommandHandler("start", start)]
    )

    app.add_handler(conv)
    print("🤖 CNN Telegram Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()

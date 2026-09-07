# """
# EcoScan — Waste Classification Flask Backend
# =============================================
# Endpoints:
#   GET  /             → Serve home page
#   GET  /upload       → Serve upload page
#   GET  /webcam       → Serve webcam page
#   POST /predict      → Accept image, return JSON prediction
# """

# import os
# import io
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from PIL import Image
# import tensorflow as tf

# # ── Config ──────────────────────────────────────────────────────────────────
# MODEL_PATH  = os.path.join("model", "best_waste_model.h5")
# IMAGE_SIZE  = (128, 128)
# CLASS_NAMES = ["Organic", "Recyclable"]   # index 0 = Organic, 1 = Recyclable

# app = Flask(__name__, template_folder="templates", static_folder="static")

# # ── Load model once at startup ───────────────────────────────────────────────
# print("[EcoScan] Loading model…")
# model = tf.keras.models.load_model(MODEL_PATH)
# print(f"[EcoScan] Model loaded from {MODEL_PATH}")


# # ── Preprocessing helper ─────────────────────────────────────────────────────
# def preprocess_image(image_bytes: bytes) -> np.ndarray:
#     """Convert raw image bytes → normalised numpy array (1, 128, 128, 3)."""
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize(IMAGE_SIZE)
#     arr = np.array(img, dtype=np.float32) / 255.0          # rescale [0,1]
#     arr = np.expand_dims(arr, axis=0)                       # add batch dim
#     return arr


# # ── Routes ───────────────────────────────────────────────────────────────────
# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/upload")
# def upload_page():
#     return render_template("upload.html")


# @app.route("/webcam")
# def webcam_page():
#     return render_template("webcam.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     Accepts: multipart/form-data with field 'file' (image).
#     Returns: JSON  { "label": "Organic"|"Recyclable",
#                      "confidence": float (0–100),
#                      "raw_score": float (sigmoid output) }
#     """
#     if "file" not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     try:
#         img_array = preprocess_image(file.read())
#         raw_score = float(model.predict(img_array, verbose=0)[0][0])

#         # sigmoid > 0.5  →  index 1 (Recyclable)
#         # sigmoid ≤ 0.5  →  index 0 (Organic)
#         if raw_score > 0.5:
#             label      = "Recyclable"
#             confidence = round(raw_score * 100, 2)
#         else:
#             label      = "Organic"
#             confidence = round((1 - raw_score) * 100, 2)

#         return jsonify({
#             "label":      label,
#             "confidence": confidence,
#             "raw_score":  round(raw_score, 4)
#         })

#     except Exception as e:
#         app.logger.error(f"Prediction error: {e}")
#         return jsonify({"error": str(e)}), 500


# # ── Run ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)





"""
EcoScan — Waste Classification Flask Backend
=============================================
Endpoints:
  GET  /             → Serve home page
  GET  /upload       → Serve upload page
  GET  /webcam       → Serve webcam page
  POST /predict      → Accept image, return JSON prediction
  POST /chat         → Accept label + question, return chatbot reply (Ollama)
"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

# ── Import chatbot ───────────────────────────────────────────────────────────
from chatbot import ask_ollama

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "best_waste_model.h5")
IMAGE_SIZE  = (128, 128)
CLASS_NAMES = ["Organic", "Recyclable"]   # index 0 = Organic, 1 = Recyclable

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load model once at startup ───────────────────────────────────────────────
print("[EcoScan] Loading model…")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[EcoScan] Model loaded from {MODEL_PATH}")


# ── Preprocessing helper ─────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes → normalised numpy array (1, 128, 128, 3)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart/form-data with field 'file' (image).
    Returns: JSON  { "label": "Organic"|"Recyclable",
                     "confidence": float (0–100),
                     "raw_score": float (sigmoid output) }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_array = preprocess_image(file.read())
        raw_score = float(model.predict(img_array, verbose=0)[0][0])

        if raw_score > 0.5:
            label      = "Recyclable"
            confidence = round(raw_score * 100, 2)
        else:
            label      = "Organic"
            confidence = round((1 - raw_score) * 100, 2)

        return jsonify({
            "label":      label,
            "confidence": confidence,
            "raw_score":  round(raw_score, 4)
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chatbot endpoint — powered by local Ollama.

    Accepts JSON:
    {
        "label":      "Organic" | "Recyclable",
        "confidence": 87.32,
        "question":   "How do I compost at home?"   ← optional
    }

    Returns JSON:
    {
        "reply": "..."
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    label      = data.get("label", "")
    confidence = float(data.get("confidence", 80.0))
    question   = data.get("question", "")

    if label not in ("Organic", "Recyclable"):
        return jsonify({"error": "label must be 'Organic' or 'Recyclable'"}), 400

    try:
        reply = ask_ollama(label, confidence, question)
        return jsonify({"reply": reply})
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
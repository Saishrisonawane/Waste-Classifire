"""
save_model.py
=============
Run this script AFTER training to save the model in formats
that Flask (app.py) and TensorFlow Serving can load.

Usage (from project root):
    python save_model.py
"""

import os
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────
KAGGLE_H5   = "/kaggle/working/best_waste_model.h5"   # where Kaggle saved it
LOCAL_DIR   = "model"                                  # local folder
H5_OUTPUT   = os.path.join(LOCAL_DIR, "best_waste_model.h5")
TF_OUTPUT   = os.path.join(LOCAL_DIR, "saved_model")  # TF SavedModel format

os.makedirs(LOCAL_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────
print(f"[save_model] Loading from: {KAGGLE_H5}")
model = tf.keras.models.load_model(KAGGLE_H5)
model.summary()

# ── Save as .h5 (used by Flask) ────────────────────────────────────────────
model.save(H5_OUTPUT)
print(f"[save_model] ✅ Saved .h5 → {H5_OUTPUT}")

# ── Save as TF SavedModel (optional, for TF Serving) ──────────────────────
model.save(TF_OUTPUT)
print(f"[save_model] ✅ Saved TF SavedModel → {TF_OUTPUT}")

# ── Quick sanity check ─────────────────────────────────────────────────────
import numpy as np
dummy = np.random.rand(1, 128, 128, 3).astype(np.float32)
loaded = tf.keras.models.load_model(H5_OUTPUT)
pred   = loaded.predict(dummy, verbose=0)
print(f"[save_model] ✅ Sanity check — raw score: {pred[0][0]:.4f}")
print("[save_model] All done! Place the 'model/' folder in your project root.")

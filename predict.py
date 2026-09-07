"""
predict.py — Single image prediction utility
=============================================
Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model model/best_waste_model.h5
"""

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH  = "model/best_waste_model.h5"
IMAGE_SIZE  = (128, 128)
CLASS_NAMES = ["Organic", "Recyclable"]


def predict(image_path: str, model_path: str = MODEL_PATH):
    model = tf.keras.models.load_model(model_path)

    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    raw = float(model.predict(arr, verbose=0)[0][0])

    if raw > 0.5:
        label      = "Recyclable"
        confidence = round(raw * 100, 2)
    else:
        label      = "Organic"
        confidence = round((1 - raw) * 100, 2)

    print(f"\n{'='*40}")
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence}%")
    print(f"  Raw score  : {raw:.4f}")
    print(f"{'='*40}\n")
    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoScan — predict a single image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .h5 model")
    args = parser.parse_args()
    predict(args.image, args.model)

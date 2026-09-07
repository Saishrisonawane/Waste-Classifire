"""
train.py — EcoScan CNN Training Script
=======================================
Trains the binary waste classifier (Organic vs Recyclable) and saves
the best model to model/best_waste_model.h5.

Usage:
    python train.py

Make sure your dataset is structured as:
    data/
      TRAIN/
        O/   (organic images)
        R/   (recyclable images)
      TEST/
        O/
        R/
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH  = os.path.join("data", "TRAIN")
TEST_PATH   = os.path.join("data", "TEST")
MODEL_DIR   = "model"
MODEL_SAVE  = os.path.join(MODEL_DIR, "best_waste_model.h5")
IMAGE_SIZE  = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 50

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data Generators ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="training", shuffle=True, seed=42
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="validation", shuffle=False, seed=42
)
test_gen = test_datagen.flow_from_directory(
    TEST_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)

print(f"Classes  : {train_gen.class_indices}")
print(f"Train    : {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

# ── Build CNN ─────────────────────────────────────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(32,  (3,3), activation="relu", padding="same", input_shape=(128, 128, 3)),
    BatchNormalization(),
    Conv2D(32,  (3,3), activation="relu", padding="same"),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    # Block 2
    Conv2D(64,  (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64,  (3,3), activation="relu", padding="same"),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    # Block 4
    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    # Classifier head
    Flatten(),
    Dense(512, activation="relu"), BatchNormalization(), Dropout(0.5),
    Dense(256, activation="relu"), Dropout(0.3),
    Dense(1,   activation="sigmoid")          # binary: Organic=0, Recyclable=1
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                      patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath=MODEL_SAVE, monitor="val_accuracy",
                    save_best_only=True, verbose=1)
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    train_gen, epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks, verbose=1
)

print(f"\n✅ Training Complete!")
print(f"Best Val Accuracy : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Epochs ran        : {len(history.history['accuracy'])}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history.history["loss"],         label="Train")
axes[1].plot(history.history["val_loss"],     label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
plt.show()

# ── Evaluate ──────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\n✅ Test Accuracy : {test_acc*100:.2f}%")
print(f"   Test Loss    : {test_loss:.4f}")

test_gen.reset()
y_pred = (model.predict(test_gen) > 0.5).astype(int).flatten()
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

print(f"\n✅ Model saved at : {MODEL_SAVE}")

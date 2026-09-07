# 🌿 EcoScan — AI Waste Classifier

A deep learning web app that classifies waste as **Organic** or **Recyclable** using a custom CNN trained on 22K+ images.

---

## 📁 Project Structure

```
waste-classifier/
│
├── app.py                  ← Flask web server (main entry point)
├── train.py                ← Model training script
├── predict.py              ← CLI prediction utility
├── save_model.py           ← Export Kaggle model to local folder
├── requirements.txt        ← All Python dependencies
├── README.md               ← This file
│
├── model/                  ← Model artifacts (created after training)
│   ├── best_waste_model.h5     ← Trained Keras model (loaded by Flask)
│   ├── saved_model/            ← TF SavedModel format (optional)
│   ├── training_curves.png     ← Accuracy/loss plots
│   └── confusion_matrix.png    ← Evaluation confusion matrix
│
├── data/                   ← Dataset (download from Kaggle)
│   ├── TRAIN/
│   │   ├── O/              ← Organic training images
│   │   └── R/              ← Recyclable training images
│   └── TEST/
│       ├── O/
│       └── R/
│
├── templates/              ← HTML pages served by Flask
│   ├── index.html          ← Home page
│   ├── upload.html         ← Upload image + predict
│   └── webcam.html         ← Live webcam capture + predict
│
├── static/                 ← Static assets
│   ├── css/
│   ├── js/
│   └── images/
│
└── notebooks/
    └── waste-classification.ipynb   ← Original Kaggle notebook
```

---

## 🚀 Quick Start

### 1. Clone / open project in VS Code
```bash
cd waste-classifier
code .
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your trained model
Copy `best_waste_model.h5` from Kaggle into the `model/` folder:
```
model/
└── best_waste_model.h5
```
Or run `save_model.py` to export it automatically.

### 5. Run the Flask server
```bash
python app.py
```
Open http://localhost:5000 in your browser. 🎉

---

## 🧠 Model Details

| Property       | Value                        |
|---------------|------------------------------|
| Architecture  | Custom CNN (4 conv blocks)   |
| Input size    | 128 × 128 × 3 (RGB)          |
| Output        | Sigmoid (binary)             |
| Classes       | Organic (0) / Recyclable (1) |
| Optimizer     | Adam (lr=0.001)              |
| Loss          | Binary Crossentropy          |
| Max Epochs    | 50 (early stopping)          |
| Val Accuracy  | ~93%                         |

### CNN Architecture
```
Input (128×128×3)
    ↓
Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    ↓
Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    ↓
Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
    ↓
Block 4: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BN → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.3)
    ↓
Dense(1, sigmoid)  →  [0 = Organic | 1 = Recyclable]
```

---

## 🌐 API Reference

### `POST /predict`

Classify a waste image.

**Request** — `multipart/form-data`
| Field  | Type | Description          |
|--------|------|----------------------|
| `file` | File | Image file (JPG/PNG) |

**Response** — `application/json`
```json
{
  "label":      "Organic",
  "confidence": 87.32,
  "raw_score":  0.1268
}
```

**Example (curl)**
```bash
curl -X POST http://localhost:5000/predict \
     -F "file=@waste_image.jpg"
```

**Example (Python)**
```python
import requests
with open("waste_image.jpg", "rb") as f:
    res = requests.post("http://localhost:5000/predict", files={"file": f})
print(res.json())
```

---

## 🔮 Training Your Own Model

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/techsash/waste-classification-data

2. Place it in `data/TRAIN/` and `data/TEST/`

3. Run training:
```bash
python train.py
```

4. The best model is saved to `model/best_waste_model.h5`

---

## 🖥️ Pages

| Page           | URL              | Description                       |
|---------------|------------------|-----------------------------------|
| Home           | `/`              | Landing page with project info    |
| Upload         | `/upload`        | Upload image → AI classification  |
| Webcam         | `/webcam`        | Live webcam → AI classification   |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — model training & inference
- **Flask** — web server & REST API
- **Pillow** — image preprocessing
- **OpenCV** — image utilities
- **NumPy** — array operations
- **HTML / CSS / Vanilla JS** — frontend (no frameworks needed)

---

## 🌍 Deployment (optional)

### Gunicorn (production)
```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

### Docker (optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

---

## 📊 Dataset

- **Source**: [Waste Classification Data — Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- **Train images**: ~22,500
- **Test images**: ~2,500
- **Classes**: `O` (Organic), `R` (Recyclable)

---

## 📝 License

MIT License — Free to use, modify, and distribute.

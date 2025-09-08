# Pneumonia Detection from Chest X-Rays (DenseNet-121, PyTorch)

⚠️ **For research & education only – not for clinical use.**

This project provides a complete deep learning pipeline for pneumonia detection from chest X-ray (CXR) images.  
It includes training, evaluation, explainability (Grad-CAM), a FastAPI backend, and a Streamlit UI.  

---

## 📥 Clone the Repository

```bash
git clone https://github.com/nitishdhamu/pneumonia-detection.git
cd pneumonia-detection
```

---

## 📂 Dataset Setup

We use the **Kaggle Chest X-Ray Images (Pneumonia)** dataset:  
🔗 [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

After downloading, extract and place it in:

```
pneumonia-detection/
  data/kaggle_chest_xray/
    train/NORMAL/...
    train/PNEUMONIA/...
    val/NORMAL/...
    val/PNEUMONIA/...
    test/NORMAL/...
    test/PNEUMONIA/...
```

---

## ⚙️ Environment Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell / CMD):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Training

Run training:

```bash
python -m backend.train
```

- Default: **DenseNet-121**, 5 epochs.  
- Checkpoints and plots are saved in `outputs/`.

---

## 📊 Evaluation

Run evaluation on the test set:

```bash
python -m backend.evaluate
```

Outputs saved in `outputs/`:
- `test_metrics.json` with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC  
- ROC and PR curves  
- Confusion matrix  
- Training history plots  

---

## 🌐 API Server

Start the FastAPI backend:

```bash
python -m backend.api
```

Endpoints:
- `GET /health` → service status  
- `POST /predict` → predict on one image  

Docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💻 Streamlit UI

In a **second terminal** (keep the API running):

```bash
streamlit run frontend/streamlit_app.py
```

Features:
- Drag & drop image upload (PNG/JPG)  
- Prediction label + confidence  
- Grad-CAM overlay visualization  
- Batch inference support  
- Metrics dashboard (loss/accuracy, ROC, PR curves)  

---

## 📊 Results & Visualization

- **Training curves**: `outputs/loss.png`, `outputs/acc.png`  
- **ROC/PR curves**: `outputs/roc.png`, `outputs/pr.png`  
- **Grad-CAM overlays**: Saved automatically during inference and displayed in Streamlit  

---

## 🎯 Project Targets

- ROC-AUC ≥ 0.90 on held-out test set  
- Grad-CAM highlights plausible lung regions  
- Lightweight structure compared to major project  

---

## 👨‍💻 Author

This project was developed by **Nitish Dhamu** as part of the Minor Project.

---

## ⚖️ Notes

- Dataset is pediatric; may not generalize to adults.  
- Folder-based labels can include some noise.  
- No PHI is collected or stored.  
- Disclaimer shown: *Not for diagnostic use*.  

---

## 📜 License

MIT

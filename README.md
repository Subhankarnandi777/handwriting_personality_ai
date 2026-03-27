# ✍️ Handwriting Personality AI

An end-to-end AI system that analyses handwriting images and predicts **Big-Five personality traits** using computer vision, handcrafted graphological features, and pretrained deep learning models.

---

## 🗂️ Project Structure

```
handwriting_personality_ai/
│
├── input/handwriting_images/    ← Drop your handwriting images here
├── output/
│   ├── results/                 ← features.json + analysis.png
│   └── reports/                 ← personality_report.txt
│
├── src/
│   ├── preprocessing/           ← Cleaning, thresholding, segmentation
│   ├── feature_extraction/      ← Slant, spacing, pressure, baseline, size, margins
│   ├── deep_features/           ← ResNet-50, ViT, feature fusion
│   ├── personality_model/       ← Rule engine, ML predictor, trait mapping
│   └── utils/                   ← Config, helpers, visualisation
│
├── models/
│   ├── pretrained/              ← resnet50.pth, vit_model.pth (auto-downloaded)
│   └── ml_models/               ← personality_model.pkl (optional trained model)
│
├── app/
│   ├── streamlit_app.py         ← Web UI
│   └── ui_utils.py
│
├── main.py                      ← CLI entry point
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Clone / unzip the project
cd handwriting_personality_ai

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option A — Command Line

```bash
# Full pipeline (with deep features)
python main.py --image input/handwriting_images/sample.jpg

# Fast mode (skip ResNet + ViT — rule engine only)
python main.py --image input/handwriting_images/sample.jpg --no-deep

# Without saving outputs
python main.py --image input/handwriting_images/sample.jpg --no-save
```

### Option B — Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 How It Works

```
Handwriting Image
       ↓
  Preprocessing        (noise removal, deskew, normalisation)
       ↓
  Thresholding         (Otsu or Adaptive — ink vs paper)
       ↓
  Segmentation         (lines → words → characters)
       ↓
  Feature Extraction
    • Slant             (Hough lines → stroke angle)
    • Spacing           (word / letter / line gaps)
    • Pressure          (ink pixel intensity analysis)
    • Baseline          (RANSAC line fitting)
    • Letter Size       (bounding box statistics)
    • Margins           (projection profiles)
       ↓
  Deep Features        (ResNet-50 2048-D + ViT 768-D embeddings)
       ↓
  Feature Fusion       (handcrafted + deep → concatenated vector)
       ↓
  Personality Model    (rule engine or ML model → Big-Five scores)
       ↓
  Report + Visualisation
```

---

## 🎭 Big-Five Personality Traits

| Trait | High Score | Low Score |
|---|---|---|
| 🎨 **Openness** | Creative, curious | Conventional, routine |
| 📋 **Conscientiousness** | Organised, reliable | Spontaneous, flexible |
| 🌟 **Extraversion** | Sociable, energetic | Reserved, introspective |
| 🤝 **Agreeableness** | Compassionate, cooperative | Competitive, direct |
| 🌊 **Neuroticism** | Anxious, sensitive | Stable, calm |

---

## 🔬 Graphological Feature Rules (Examples)

| Feature | Effect |
|---|---|
| Right slant | ↑ Extraversion, ↑ Agreeableness |
| Heavy pressure | ↑ Conscientiousness, ↑ Neuroticism |
| Large letters | ↑ Extraversion, ↑ Openness |
| Consistent size | ↑ Conscientiousness |
| Straight baseline | ↑ Conscientiousness, ↓ Neuroticism |
| Wide word spacing | ↑ Openness, ↑ Conscientiousness |
| Variable strokes | ↑ Openness (artistic) |

---

## 📤 Outputs

| File | Location | Contents |
|---|---|---|
| `features.json` | `output/results/` | All numeric features |
| `analysis.png` | `output/results/` | 4-panel visualisation |
| `report.txt` | `output/reports/` | Full text report + rules |

---

## 🔧 Configuration

All parameters are in `src/utils/config.py`:

```python
PREPROCESS["threshold_method"]   # "otsu" or "adaptive"
PREPROCESS["resize_width"]        # normalise image width
DEEP["device"]                    # "cpu" or "cuda"
PERSONALITY["use_rule_engine"]    # True = always use rules
```

---

## 📦 Dependencies

```
opencv-python      scikit-learn       streamlit
numpy              torch              matplotlib
pandas             torchvision        pillow
scikit-image       transformers       scipy
joblib             fpdf2              plotly
```

---

## 📝 Notes

- **No training dataset required** — the system uses pretrained ImageNet models + graphology rules.
- Deep learning models are **auto-downloaded** on first run and cached in `models/pretrained/`.
- An optional `personality_model.pkl` can be placed in `models/ml_models/` to replace the rule engine with a trained scikit-learn model.
- For best results, use **clear scans** of handwriting on white paper, minimum 300 DPI.
# handwriting_personality_ai

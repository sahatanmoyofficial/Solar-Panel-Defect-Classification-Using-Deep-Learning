# ☀️ Solar Panel Defect Classification Using Deep Learning & Transfer Learning (SPDCL)

> **6-class solar panel defect detection from images — benchmarking custom CNNs against MobileNetV2 and EfficientNetB0 transfer learning, with KerasTuner HPO achieving 83.05% validation accuracy, deployed as a Streamlit web app with top-3 predictions and full class probability display**
>
> A complete Computer Vision research-to-deployment pipeline: 885 images across 6 defect classes → class weight computation for imbalanced data → 5 progressive experiments revealing why scratch-trained CNNs overfit → EfficientNetB0 + KerasTuner (20 trials, 6h 49m) → `trained_effnet_finetune.h5` → Streamlit app with EfficientNet-specific preprocessing, confidence scoring, and Clean/Defective status alerts.

---

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)](https://tensorflow.org/)
[![EfficientNetB0](https://img.shields.io/badge/Model-EfficientNetB0-brightgreen)](https://keras.io/api/applications/efficientnet/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![KerasTuner](https://img.shields.io/badge/HPO-KerasTuner-yellow)](https://keras.io/keras_tuner/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1WCw9LyIt3XQDKxPOSk4YBauHaLzWDOuU/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Problem Statement](#1-problem-statement) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [System Architecture](#4-system-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Dataset & Class Imbalance](#6-dataset--class-imbalance) |
| 7 | [5-Experiment Progression](#7-5-experiment-progression) |
| 8 | [EfficientNetB0 + KerasTuner HPO](#8-efficientnetb0--kerastuner-hpo) |
| 9 | [Results Summary](#9-results-summary) |
| 10 | [Streamlit Web Application](#10-streamlit-web-application) |
| 11 | [How to Replicate](#11-how-to-replicate) |
| 12 | [Business Applications & Other Domains](#12-business-applications--other-domains) |
| 13 | [How to Improve This Project](#13-how-to-improve-this-project) |
| 14 | [Troubleshooting](#14-troubleshooting) |
| 15 | [Glossary](#15-glossary) |

---

## 1. Problem Statement

### What problem are we solving?

Solar energy installations generate electricity at scale, but panel performance degrades significantly when panels are covered with bird droppings, dust, snow, or suffer physical or electrical damage. Manual inspection of large solar farms requires field personnel, is time-consuming, and often delayed — meaning defects go undetected for weeks or months, reducing energy output.

Automated visual defect classification from drone or camera images enables:
- **Continuous monitoring** — daily or weekly inspection of all panels without manual effort
- **Prioritised maintenance** — dispatch field teams only to panels flagged as defective
- **Energy yield optimisation** — early detection of performance-reducing contamination
- **Failure prevention** — identify electrical damage before cascading failures occur

### The 6 Classes

| Class | Index | Description | Impact |
|-------|-------|-------------|--------|
| **Bird-drop** | 0 | Bird droppings on panel surface | Localised shading → hot spots → cell damage |
| **Clean** | 1 | No defects — normal operation | Baseline: maximum energy production |
| **Dusty** | 2 | Dust/soil accumulation | Uniform reduction in light transmission |
| **Electrical-damage** | 3 | Electrical faults, burn marks | Critical — immediate maintenance required |
| **Physical-Damage** | 4 | Cracks, broken glass | Critical — structural integrity compromised |
| **Snow-Covered** | 5 | Snow obscuring the panel | Temporary — clears with weather or cleaning |

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Task** | 6-class image classification (solar panel defect type) |
| **Total images** | 885 (imbalanced across classes) |
| **Train / Val split** | 80/20 → 708 train / 177 validation |
| **Input shape** | 224 × 224 × 3 (RGB) |
| **Imbalance handling** | Class weights computed per class |
| **Experiment 1** | Custom CNN 3-blocks (11.17M params) — severe overfitting |
| **Experiment 2** | Custom CNN + BN + Dropout — exploding loss |
| **Experiment 3** | Custom CNN + Grok augmentation recs — still failed |
| **Experiment 4** | MobileNetV2 Transfer Learning — ~63% val |
| **Experiment 5** | EfficientNetB0 Transfer Learning — ~83% val |
| **KerasTuner best** | **83.05% val accuracy** (20 trials, 6h 49m) |
| **Best HPO params** | rotation=0.05, zoom=0.05, dropout=0.4, dense_units=128, lr=0.00291 |
| **Deployed model** | EfficientNetB0 + best weights (`trained_effnet_finetune.h5`) |
| **Preprocessing** | `efficientnet.preprocess_input` (NOT simple /255) |
| **Serving** | Streamlit — upload, top-3 predictions, full class probabilities, status alert |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.12 | Core language |
| **Deep Learning** | TensorFlow / Keras | Model definition, training, inference |
| **Base models** | MobileNetV2, EfficientNetB0 | Pre-trained ImageNet weights |
| **HPO** | KerasTuner 1.4.8 | RandomSearch over augmentation, dropout, dense units, LR |
| **Data pipeline** | `tf.keras.utils.image_dataset_from_directory` | Batched loading with 80/20 split |
| **Data augmentation** | Keras `RandomFlip`, `RandomRotation`, `RandomZoom` | In-model layers, applied only at training |
| **Preprocessing** | `EfficientNetB0.preprocess_input` | EfficientNet-specific normalisation |
| **Imbalance** | `class_weight` dict in `model.fit()` | Weights inversely proportional to class frequency |
| **Image handling** | PIL `Image` | At inference: open, convert RGB, resize |
| **Web framework** | Streamlit | Upload → predictions → Clean/Defective alert |
| **Model caching** | `@st.cache_resource` | Load model once across all sessions |
| **Serialisation** | `.h5` (Keras legacy format) | `model.save()` — includes architecture + weights |
| **Training env** | Google Colab (GPU) | 1 GPU available |

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                   RESEARCH PHASE (Colab)                             │
│                                                                      │
│  885 images, 6 classes, imbalanced                                  │
│  80/20 split: 708 train / 177 val                                   │
│                                                                      │
│  Class weights (inverse frequency):                                  │
│    Bird-drop: 0.753 · Clean: 0.749 · Dusty: 0.765                  │
│    Electrical: 1.411 · Physical: 2.106 · Snow: 1.182               │
│                                                                      │
│  Experiment 1: Custom CNN (3 blocks, 11.17M params)                 │
│    → 98% train / 57% val — catastrophic overfitting                 │
│                                                                      │
│  Experiment 2: Custom CNN + BN + Dropout                            │
│    → exploding loss (loss 25→16→20) — training failed              │
│                                                                      │
│  Experiment 3: CNN + augmentation (Grok rec)                        │
│    → 9% val accuracy — worse than random                            │
│                                                                      │
│  Experiment 4: MobileNetV2 TL (frozen base)                         │
│    → 63% peak val accuracy                                          │
│                                                                      │
│  Experiment 5: EfficientNetB0 TL (frozen base)                      │
│    → 83% peak val accuracy                                          │
│                                                                      │
│  KerasTuner HPO (20 trials × EfficientNetB0)                        │
│    → 83.05% best val accuracy                                       │
│    Params: rotation=0.05, zoom=0.05, drop=0.4, dense=128, lr=0.003 │
│    → save trained_effnet_finetune.h5                                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│               PRODUCTION APP (app.py)                                │
│                                                                      │
│  @st.cache_resource → tf.keras.models.load_model(.h5)               │
│                                                                      │
│  Upload .jpg/.jpeg/.png                                              │
│  → PIL.Image.open().convert('RGB')                                   │
│  → resize to 224×224                                                 │
│  → np.array → expand_dims → preprocess_input (EfficientNet-specific)│
│  → model.predict → predictions[0] shape (6,)                        │
│                                                                      │
│  Display:                                                            │
│    • Predicted class (1st) + confidence %                           │
│    • ✅ "Clean" or ⚠️ "Defect detected" alert                       │
│    • Top 3 predictions ranked                                        │
│    • Expandable: all 6 class probabilities                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Repository Structure

```
Solar-Panel-Defect-Classification/
│
├── app.py                              # Streamlit app — load model + inference + UI
├── Solar_Panel_Classification.ipynb   # Full research notebook (5 experiments + HPO)
├── requirements.txt                   # Dependencies (empty — add manually, see below)
├── Deployment.txt                     # EC2 deployment commands
└── LICENSE                            # MIT License
```

> **Note:** `trained_effnet_finetune.h5` (the saved model) must be present in the working directory for `app.py` to run. It is not included in the repository due to file size.

> **Note:** `requirements.txt` is empty — add `streamlit tensorflow pillow numpy` manually.

---

## 6. Dataset & Class Imbalance

### Dataset

| Property | Detail |
|----------|--------|
| **Total images** | 885 |
| **Classes** | 6 (subdirectory names auto-detected by Keras) |
| **Train / Val** | 708 / 177 (80/20, seed=42) |
| **Input size** | 224 × 224 × 3 |
| **Batch size** | 32 |

### Class Distribution and Weights

The dataset is imbalanced — Physical-Damage images are significantly rarer than Bird-drop, Clean, and Dusty:

| Class | Index | Count (approx.) | Class Weight |
|-------|-------|-----------------|-------------|
| **Bird-drop** | 0 | ~196 | 0.753 |
| **Clean** | 1 | ~197 | 0.749 |
| **Dusty** | 2 | ~193 | 0.765 |
| **Electrical-damage** | 3 | ~105 | 1.411 |
| **Physical-Damage** | 4 | ~70 | 2.106 |
| **Snow-Covered** | 5 | ~124 | 1.182 |

Class weights are computed as:

```python
class_weights[index] = total_images / (num_classes × class_counts[class_name])
```

Physical-Damage gets a weight of **2.106** — meaning misclassifying a Physical-Damage panel costs the model 2.1× as much as misclassifying a Clean panel. This focuses training on the rare but critical defect classes.

---

## 7. 5-Experiment Progression

### Experiment 1 — Custom CNN Baseline (Overfitting)

```
Input → Rescaling(1/255) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
      → Conv2D(128) → MaxPool → Flatten → Dense(128) → Dense(6, softmax)

Total params: 11,169,734 (42.61 MB) — all trainable
```

**Results (10 epochs):**

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|---------|
| 1 | 21.3% | 31.1% | 1.73 |
| 2 | 36.5% | 45.8% | 1.48 |
| 3 | 51.8% | 52.5% | 1.33 |
| ... | learning | learning | ... |

Then continued for 20 more epochs: **train accuracy → 99%, val accuracy → 57–60%**.

This extreme train/val gap is classic overfitting — 11.17M parameters vs only 708 training images means the model memorises the training set completely rather than learning generalisable features.

---

### Experiment 2 — Custom CNN + BN + Dropout (Failed)

Added `BatchNormalization()` and `Dropout(0.3)` after each conv block, plus `EarlyStopping`, `LearningRateScheduler`.

**Result:** Exploding loss — val_loss reached 16 → 20 by epoch 3 (~17% val accuracy). The combination of BatchNorm with poor rescaling compatibility caused training instability.

---

### Experiment 3 — Augmented CNN (Grok Recommendations, Failed)

Applied in-model data augmentation (`RandomFlip`, `RandomRotation(0.15)`, `RandomZoom(0.15)`) + `ReduceLROnPlateau` + L2 regularisation.

**Result:** Worse than random — **9% val accuracy** through all epochs. The model learned to predict a single class.

---

### Experiment 4 — MobileNetV2 Transfer Learning

```
Input → Augmentation → Rescaling(1/255) → MobileNetV2 (frozen) 
      → GlobalAveragePooling2D → Dense(128, relu) → Dense(6, softmax)

Total: 2,422,726 params  ·  Trainable: 164,742 (643 KB)
```

**Results (20 epochs):** Peak ~63% val accuracy. Immediate improvement from ImageNet features, but the simpler MobileNetV2 backbone struggled with the visual complexity of defect types (bird droppings vs. burn marks vs. cracks vs. dust).

---

### Experiment 5 — EfficientNetB0 Transfer Learning (Best Architecture)

EfficientNetB0 uses compound scaling — balancing depth, width, and resolution — producing richer features than MobileNetV2 for fine-grained visual classification:

```
Input → Augmentation → EfficientNetB0 (frozen, imagenet)
      → GlobalAveragePooling2D → Dense(128, relu) → Dense(6, softmax)

Total: 4,214,313 params  ·  Trainable: 164,742 (643 KB)
```

**Key difference from MobileNetV2:** EfficientNetB0 uses its own `preprocess_input` function — it does NOT use simple /255 normalisation. The EfficientNet preprocessing applies a specific mean/variance normalisation tuned for the ImageNet training distribution. The Streamlit app correctly applies `efficientnet.preprocess_input` at inference.

**Results (15 epochs):**

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 45.4% | 63.8% |
| 2 | 76.3% | 72.3% |
| 3 | 79.1% | 70.1% |
| ... | improving | ~83% peak |

---

## 8. EfficientNetB0 + KerasTuner HPO

### Search Space

```python
def build_model(hp):
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    # Tunable: data augmentation intensity
    rotation = hp.Float('rotation_factor', 0.0, 0.3)
    zoom     = hp.Float('zoom_factor', 0.0, 0.3)

    # Tunable: regularisation
    dropout  = hp.Float('dropout_rate', 0.2, 0.5)

    # Tunable: head capacity
    units    = hp.Int('dense_units', 64, 512, step=64)

    # Tunable: learning rate
    lr       = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
```

### Search Results

| Metric | Value |
|--------|-------|
| **Trials completed** | 20 |
| **Total search time** | 6 hours 49 minutes |
| **Best val_accuracy** | **83.05%** |
| **Best rotation_factor** | 0.05 |
| **Best zoom_factor** | 0.05 |
| **Best dropout_rate** | 0.40 |
| **Best dense_units** | 128 |
| **Best learning_rate** | 0.002910 |

### Why These Hyperparameters Won

- **rotation=0.05, zoom=0.05** — very light augmentation. Solar panel defect images have consistent orientation (panels are flat, mounted at fixed angles) — aggressive rotation/zoom removes meaningful spatial structure rather than helping generalisation
- **dropout=0.40** — moderate dropout prevents co-adaptation in the dense head without underfitting the limited 708 training examples
- **dense_units=128** — matching the original architecture; larger heads tend to overfit with only 885 total images
- **lr=0.00291** — higher than the typical 1e-3 Adam default, likely beneficial because EfficientNet's BatchNorm layers need slightly more aggressive gradient updates during the initial frozen-base head training

The best model was saved as `trained_effnet_finetune.h5`.

---

## 9. Results Summary

| Experiment | Val Accuracy | Notes |
|-----------|-------------|-------|
| Custom CNN (10 ep) | **~52%** | Barely learning |
| Custom CNN continued (20 ep) | **98% train / 57% val** | Catastrophic overfitting |
| Custom CNN + BN + Dropout | **17%** | Exploding loss — failed |
| CNN + Augmentation (Grok) | **9%** | Worse than random |
| MobileNetV2 TL (20 ep) | **~63%** | ImageNet features help |
| EfficientNetB0 TL (15 ep) | **~83%** | Compound scaling advantage |
| **EfficientNetB0 + KerasTuner** | **83.05%** | **Deployed model** |

### Critical Lessons from This Experiment Sequence

1. **11M params + 708 images = guaranteed overfitting** — the training set is simply too small for a scratch CNN of this size
2. **BatchNorm + poor LR setup = exploding loss** — training instability in Experiment 2 illustrates why BN requires careful learning rate tuning
3. **Aggressive augmentation can hurt** — when defect images have consistent spatial structure (panels face one direction), heavy rotation/zoom destroys the signal
4. **EfficientNetB0 > MobileNetV2 for fine-grained defects** — compound scaling provides richer texture and edge features needed to distinguish burn marks from bird droppings from dust
5. **EfficientNet preprocessing matters** — must use `preprocess_input` (not /255) at both training and inference

---

## 10. Streamlit Web Application

### `app.py` Features

The app loads the full saved model (architecture + weights in `.h5`), processes uploaded images with EfficientNet-specific preprocessing, and provides a rich prediction UI:

```python
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_effnet_finetune.h5")
    return model

CLASSES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered"]
```

### Inference Pipeline

```python
# 1. Open and convert
image = Image.open(uploaded_file).convert('RGB')

# 2. Resize to EfficientNet input size
img = image.resize((224, 224))

# 3. Array + EfficientNet-specific preprocessing (NOT /255!)
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array.astype(np.float32))

# 4. Predict
predictions = model.predict(img_array, verbose=0)
predicted_idx = np.argmax(predictions[0])
confidence = predictions[0][predicted_idx]

# 5. Status logic
if predicted_class == "Clean":
    st.success("The panel appears to be in good condition!!")
else:
    st.warning("A defect or contamination has been detected!!")
```

### App Features

| Feature | Implementation |
|---------|---------------|
| **File upload** | `st.file_uploader(type=["jpg","jpeg","png"])` |
| **Image display** | `st.image(image, use_column_width=True)` |
| **Primary prediction** | Top class + confidence % in bold |
| **Status alert** | `st.success` (Clean) or `st.warning` (Defect) |
| **Top 3 ranked** | `np.argsort(predictions[0])[-3:][::-1]` — ranked descending |
| **All probabilities** | `st.expander("View all class probabilities")` — expandable |
| **Model caching** | `@st.cache_resource` — single load per deployment |
| **Loading spinner** | `with st.spinner("Loading model...")` during `load_model()` |

### Running the App

```bash
# Requires trained_effnet_finetune.h5 in same directory
streamlit run app.py
# Opens http://localhost:8501
```

---

## 11. How to Replicate

### Prerequisites

- Python 3.12+
- `trained_effnet_finetune.h5` in the working directory

---

### Step 1 — Install Dependencies

```bash
pip install streamlit tensorflow pillow numpy
```

---

### Step 2 — Run the App

```bash
streamlit run app.py
```

---

### Retrain in Colab

1. Mount Google Drive with dataset at `Krish AI/Solar Panel Classification/`
2. Open `Solar_Panel_Classification.ipynb`
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run all cells in order — CNN experiments → MobileNetV2 → EfficientNetB0 → KerasTuner
5. The saved `trained_effnet_finetune.h5` in Drive can be downloaded and placed alongside `app.py`

---

### EC2 Deployment (Production)

From `Deployment.txt`:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git python3-pip python3-venv -y

git clone https://github.com/sahatanmoyofficial/Solar-Panel-Defect-Classification.git
cd Solar-Panel-Defect-Classification

python3 -m venv .venv && source .venv/bin/activate
pip install streamlit tensorflow pillow numpy

# Place trained_effnet_finetune.h5 in this directory
nohup python3 -m streamlit run app.py &
```

---

## 12. Business Applications & Other Domains

### Primary Use Cases

| Stakeholder | Application |
|-------------|------------|
| **Solar farm operators** | Drone-based daily inspection of thousands of panels — flag defective ones for maintenance teams |
| **Energy utilities** | Performance monitoring dashboards — correlate defect type with energy output loss |
| **Insurance providers** | Automated damage assessment for weather events (snow, physical damage) |
| **Solar installers** | Post-installation quality control — identify faulty panels before system goes live |
| **Research labs** | Track dust accumulation patterns by geography and season |

### Energy Impact by Defect Class

| Class | Estimated Output Reduction | Priority |
|-------|--------------------------|----------|
| **Physical-Damage** | Up to 100% | 🔴 Immediate |
| **Electrical-damage** | Up to 100% | 🔴 Immediate |
| **Bird-drop** | 5–15% (localised hot spots) | 🟡 Moderate |
| **Dusty** | 5–25% depending on severity | 🟡 Moderate |
| **Snow-Covered** | 100% (temporary) | 🟢 Weather-dependent |
| **Clean** | 0% | ✅ No action needed |

### The Architecture Generalises

| Domain | Classes | Why EfficientNetB0 Works |
|--------|---------|--------------------------|
| **PCB defect detection** | Scratch, Short, Missing hole, Open | Fine-grained texture differences |
| **Concrete crack detection** | Hairline, Wide, Structural | Subtle visual patterns |
| **Agricultural leaf disease** | 15+ diseases per crop | Small dataset, diverse classes |
| **Industrial surface defects** | Scratch, Dent, Contamination | Manufacturing QC |

---

## 13. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Fine-tune EfficientNetB0 top layers** | 🔴 High | Unfreeze top 20 layers with lr ~1e-5 after head training — expected to push from 83% to 88–92% |
| **Add confusion matrix** | 🔴 High | 83% overall accuracy may hide poor recall on Physical-Damage (class weight 2.106) — `classification_report` per class is essential |
| **Try EfficientNetB3/B4** | 🟡 Medium | Larger EfficientNet variants have richer features for fine-grained 6-class defect detection; B0 is the lightest |
| **Increase dataset size** | 🟡 Medium | 885 images across 6 classes is minimal — target 1,000+ per class via web scraping or synthetic augmentation |
| **Add GradCAM visualisation** | 🟡 Medium | Show which panel regions drove the prediction — critical for operator trust in defect classification |

### 🏗️ Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **Populate requirements.txt** | Currently empty — add `streamlit tensorflow pillow numpy` with pinned versions |
| **Add confidence threshold** | Show "Low confidence — uncertain" when `max(predictions[0]) < 0.65` |
| **Use `.keras` format** | `model.save('model.keras')` instead of `.h5` — avoids legacy format warnings |
| **Severity-coded display** | Colour-code class predictions: red for Physical/Electrical, amber for Bird-drop/Dusty, green for Clean |
| **Batch inference endpoint** | Process entire folder of drone images at once for solar farm operators |

---

## 14. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `FileNotFoundError: trained_effnet_finetune.h5` | Model file must be in same directory as `app.py` — retrain via notebook or download from project Drive |
| Model predicts wrong class consistently | Verify `preprocess_input` from `efficientnet` is being applied — using `/255` instead will degrade accuracy |
| `use_column_width` deprecation warning | Replace with `use_container_width=True` in `st.image()` |
| Low confidence on all predictions | Model may not have loaded correctly — check `@st.cache_resource` and clear Streamlit cache |
| `requirements.txt` empty — install fails | Run: `pip install streamlit tensorflow pillow numpy` manually |
| Exploding loss in Experiment 2 reproduction | This is a known training instability from the BN + high initial LR combination — it is expected and part of the research history |

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **EfficientNetB0** | A CNN architecture that uses compound scaling — simultaneously scaling network depth, width, and input resolution — achieving strong accuracy per parameter |
| **Compound scaling** | EfficientNet's key innovation: scaling all three dimensions (depth, width, resolution) together using a fixed ratio, rather than arbitrarily scaling one |
| **`preprocess_input` (EfficientNet)** | EfficientNet-specific input normalisation that applies channel-wise mean subtraction and scaling — distinct from simple `/255` rescaling used by MobileNetV2 |
| **Class weights** | Training parameter that multiplies the loss contribution of each class — higher weight for minority classes like Physical-Damage (2.106×) |
| **KerasTuner** | Keras-native HPO library — defines search spaces and runs trials to find optimal model configurations |
| **RandomSearch (KerasTuner)** | Randomly samples hyperparameter combinations for N trials — efficient for large search spaces |
| **`@st.cache_resource`** | Streamlit decorator caching a shared resource across all sessions — model loaded once per deployment |
| **`image_dataset_from_directory`** | TF utility that automatically infers class labels from subdirectory names and creates batched datasets |
| **EarlyStopping** | Callback that halts training when a monitored metric stops improving (within `patience` epochs) |
| **ReduceLROnPlateau** | Callback that reduces the learning rate by a factor when a metric plateaus |
| **Exploding loss** | Training instability where loss grows rapidly — often caused by poor learning rate + BatchNorm interaction |
| **Overfitting** | Model memorises training data rather than learning generalisable features — manifests as high train accuracy, low val accuracy |
| **`np.argsort(...)[-3:][::-1]`** | NumPy idiom: sort ascending → take last 3 (highest) → reverse → top 3 descending |
| **Sparse categorical crossentropy** | Loss for integer-labelled multi-class problems (labels as integers, not one-hot vectors) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

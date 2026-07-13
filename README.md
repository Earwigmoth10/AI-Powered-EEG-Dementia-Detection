# 🧠 AI-Powered EEG Dementia Detection System

An end-to-end machine learning web application that analyzes resting-state EEG recordings to assist in distinguishing between **Alzheimer's Disease (AD)**, **Frontotemporal Dementia (FTD)**, and **Healthy Control (HC)** subjects — built with a Flask backend, a scikit-learn/XGBoost inference pipeline, and a responsive vanilla JS/HTML/CSS frontend.

> ⚠️ **Disclaimer:** This project is a university AI research project and decision-support prototype only. It is **not** a certified medical device and must **not** be used for real clinical diagnosis.

---

## 📸 Preview

<!--
Add your own screenshots to a folder named `assets/screenshots/` in the repo root,
then update the image paths below to match your filenames.
-->

| Home / Hero | Prediction Dashboard | Results View |
|:---:|:---:|:---:|
| ![Home page](assets/screenshots/home.png) | ![Prediction dashboard](assets/screenshots/prediction.png) | ![Results](assets/screenshots/results.png) |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Model Details](#-model-details)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

This system applies machine learning to raw EEG signals to support **non-invasive, faster screening** for dementia-related neurodegeneration. Users upload an EEG recording through the web dashboard, pick a classification mode, and receive back a prediction, a confidence score, and a full class-probability breakdown — all rendered in an interactive, chart-backed results panel.

Two prediction modes are supported:

| Mode | Classes | Description |
|---|---|---|
| **Binary Classification** | `AD` vs `HC` | Distinguishes Dementia vs. Healthy Control |
| **Three-Class Classification** | `AD` vs `FTD` vs `HC` | Separates Alzheimer's Disease, Frontotemporal Dementia, and Healthy Control |

---

## 🗂 Dataset

This project is trained on the **[OpenNeuro ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.9)** dataset:

> **"A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG"**
> Miltiadous, A., Tzimourta, K. D., Afrantou, T., et al. (2023). *Data*, 8(6), 95.

**Dataset summary:**
- **88 subjects** total, resting-state, eyes-closed EEG recordings
- **36** diagnosed with Alzheimer's Disease (AD)
- **23** diagnosed with Frontotemporal Dementia (FTD)
- **29** healthy control subjects (HC/CN)
- Recorded with a **Nihon Kohden EEG 2100** clinical device
- **19 scalp electrodes** (10–20 international system: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz) + 2 mastoid reference electrodes
- Cognitive status scored via **Mini-Mental State Examination (MMSE)**
- Preprocessed with Artifact Subspace Reconstruction (ASR) and ICA-based artifact rejection (EEGLAB), distributed in BIDS format
- License: **CC0**

If you use this dataset, please cite the original paper (see [Citation](#-citation)).

---

## ✨ Features

- 🧠 **EEG Signal Analysis** — band-power feature extraction (delta, theta, alpha, beta) per channel, per epoch
- ⚡ **Fast AI Prediction** — inference completes in seconds via a pre-trained scikit-learn/XGBoost pipeline
- 📊 **Confidence & Probability Breakdown** — every prediction returns a calibrated confidence score plus per-class probabilities
- 🏥 **Clinical-style Decision Support** — plain-language disease descriptions alongside each prediction
- 📄 **Downloadable PDF Report** — generate and export a report of the prediction results (via jsPDF)
- 📱 **Responsive, Accessible UI** — mobile-friendly frontend with semantic HTML and ARIA labeling

---

## ⚙️ How It Works

```
 1. Upload EEG File          →  User uploads a .set (EEGLAB) recording via the dashboard
 2. Select Prediction Model  →  Binary (AD vs HC) or Three-Class (AD vs FTD vs HC)
 3. AI Analysis              →  Backend preprocesses signal → extracts band-power features
                                 → scales/selects features → runs trained classifier
 4. Prediction Results       →  Classification label, confidence %, per-class probabilities,
                                 and a short clinical description are returned to the UI
```

**Three-class pipeline (production, real inference):**

```
EEGLAB .set file
   → mne.io.read_raw_eeglab()
   → band-pass filter (0.5–30 Hz)
   → fixed-length 4s epochs
   → Welch PSD per epoch/channel
   → band power (delta/theta/alpha/beta) + slowing ratio per channel  → 95 features/epoch
   → StandardScaler → Feature Selector → trained classifier (predict_proba)
   → epoch-level probabilities averaged into one patient-level prediction
```

---

## 🗃 Project Structure

```
week 1/
├── app.py                              # Flask app: routes, request/response handling
├── predict.py                          # Production inference pipeline (feature extraction + model calls)
├── model_utils.py                      # Reference/legacy model-loading scaffold
├── utils.py                            # File validation & safe upload helpers
├── requirements.txt                    # Python dependencies
├── Alzheimer's.py                      # Exploratory / training script
│
├── models/                             # Trained model artifacts (joblib .pkl)
│   ├── production_eeg_model.pkl               # Binary classifier
│   ├── production_scaler.pkl                  # Binary scaler
│   ├── production_eeg_model_3class.pkl        # Three-class classifier
│   ├── production_scaler_3class.pkl           # Three-class scaler
│   ├── production_selector_3class.pkl         # Three-class feature selector
│   └── production_label_encoder_3class.pkl    # Three-class label encoder
│
├── gsp-alzheimer-detection-main/       # Model training & experimentation
│   ├── main.ipynb                          # Training notebook
│   ├── features_epoched_v2.csv             # Extracted feature dataset
│   ├── heldout_test.pkl                    # Held-out test split
│   └── README.md
│
├── static/                             # Frontend assets
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html                          # Main dashboard UI (Jinja2 template)
│
├── dataset/                            # Raw/processed EEG dataset (OpenNeuro ds004504)
├── uploads/                            # Runtime folder for user-uploaded EEG files
├── server open.pdf                     # Project documentation / write-up
└── __pycache__/
```

---

## 🛠 Tech Stack

**Backend**
- [Flask](https://flask.palletsprojects.com/) + Flask-CORS
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- [MNE-Python](https://mne.tools/) — EEG signal reading & preprocessing
- [joblib](https://joblib.readthedocs.io/), NumPy, pandas, SciPy

**Frontend**
- HTML5, CSS3, vanilla JavaScript
- [Chart.js](https://www.chartjs.org/) — probability/confidence visualizations
- [jsPDF](https://github.com/parallax/jsPDF) — downloadable PDF reports

**Data**
- [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.9)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Keep this terminal window **open and running** — the server needs to stay active.

### 5. Open in your browser

Navigate to:

```
http://127.0.0.1:5000
```

> On Windows, from the project directory this looks like:
> ```
> cd C:\Users\<you>\Downloads\week 1
> python app.py
> ```
> then visiting `http://127.0.0.1:5000` in your browser.

---

## 📡 API Reference

### `POST /predict`

Runs a prediction on an uploaded EEG file.

**Form-data parameters:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | ✅ | EEG recording (`.set`, `.edf`, `.csv`, `.mat`, `.txt`, `.npy`) |
| `model` | string | ✅ | `binary` or `three-class` |

**Example (`curl`):**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@sample_recording.set" \
  -F "model=three-class"
```

**Success response `200`:**

```json
{
  "status": "Success",
  "prediction": "Frontotemporal Dementia",
  "confidence": 88.2,
  "model": "Three-Class Classification",
  "probabilities": { "AD": 5.1, "FTD": 88.2, "HC": 6.7 },
  "description": "Frontotemporal dementia primarily affects the frontal and temporal lobes, impacting behavior, personality, and language."
}
```

**Error response (`400` / `422` / `500` / `503`):**

```json
{
  "status": "Error",
  "prediction": null,
  "confidence": null,
  "model": null,
  "probabilities": {},
  "description": "",
  "message": "Descriptive error message here."
}
```

---

## 🤖 Model Details

| Pipeline | Input Features | Status |
|---|---|---|
| **Binary (AD vs HC)** | 36-dimensional feature vector | ⚠️ Uses placeholder/mock features — see limitations below |
| **Three-Class (AD/FTD/HC)** | 95 band-power features (19 channels × 5 features) extracted per 4s epoch via Welch PSD | ✅ Real, notebook-derived pipeline |

Both pipelines use scikit-learn-compatible classifiers (trained in `gsp-alzheimer-detection-main/main.ipynb`), serialized with `joblib`, and loaded once at app startup for fast repeated inference.

---

## ⚠️ Known Limitations

- **Binary pipeline uses mock features.** The graph-signal-processing feature extraction used during binary-model training was not ported into `predict.py`; `extract_mock_features_binary()` currently returns placeholder values. **Do not trust binary-mode predictions** until this is replaced with the real feature pipeline.
- The three-class pipeline currently only accepts **EEGLAB `.set`** files for real inference; other accepted extensions (`.csv`, `.edf`, `.mat`, `.txt`, `.npy`) are validated on upload but do not yet have matching feature-extraction code paths.
- This tool is a research/decision-support prototype and has **not** been clinically validated — it should never be used as a standalone diagnostic tool.

---

## 🗺 Roadmap

- [ ] Port real feature-extraction pipeline into `extract_mock_features_binary()`
- [ ] Add support for `.edf`, `.csv`, `.mat`, `.npy` feature extraction paths
- [ ] Add authentication for clinical/research deployments
- [ ] Containerize with Docker for easier deployment
- [ ] Add automated tests for the inference pipeline

---

## 📚 Citation

If you use this project or the underlying dataset, please cite:

```bibtex
@article{miltiadous2023dataset,
  title={A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG},
  author={Miltiadous, Andreas and Tzimourta, Katerina D. and Afrantou, Theodora and Ioannidis, Panagiotis and Grigoriadis, Nikolaos and Tsalikakis, Dimitrios G. and Angelidis, Pantelis and Tsipouras, Markos G. and Glavas, Euripidis and Giannakeas, Nikolaos and Tzallas, Alexandros T.},
  journal={Data},
  volume={8},
  number={6},
  pages={95},
  year={2023},
  publisher={MDPI}
}
```

Dataset DOI: [10.18112/openneuro.ds004504](https://doi.org/10.18112/openneuro.ds004504)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). The underlying dataset (OpenNeuro ds004504) is released under **CC0**.

---

## 📬 Contact

Developed as a university Artificial Intelligence project.

For questions, collaboration, or academic inquiries, feel free to open an issue or reach out via the contact form in the app.

---

<p align="center"><sub>Built with Flask, scikit-learn, MNE-Python, and a lot of EEG epochs 🧠⚡</sub></p>

"""
model_utils.py
==================================================================
Model loading and inference utilities for the EEG Dementia AI
backend.

This module is intentionally kept separate from app.py so the
Flask routing/HTTP layer stays clean and the machine-learning
code stays reusable and testable on its own.

>>> WHERE TO PLUG IN YOUR TRAINED MODELS <<<
Every place you need to edit is marked with a
    # TODO: LOAD YOUR MODEL HERE
or
    # TODO: RUN YOUR MODEL INFERENCE HERE
comment block below. Nothing in this file invents or hardcodes
a prediction — until you wire in your real models, the inference
functions raise ModelNotLoadedError on purpose.
==================================================================
"""

import os
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Filenames are placeholders — rename to match whatever you exported
# from your training notebook (e.g. "binary_model.h5", "binary_model.pkl",
# "binary_model.pt", etc.)
BINARY_MODEL_PATH = os.path.join(MODELS_DIR, "binary_model.pkl")
THREE_CLASS_MODEL_PATH = os.path.join(MODELS_DIR, "three_class_model.pkl")

# Class label ordering must match the order your model was trained with.
BINARY_CLASSES = ["AD", "HC"]
THREE_CLASS_CLASSES = ["AD", "FTD", "HC"]

# Human-readable labels returned to the frontend.
CLASS_DISPLAY_NAMES = {
    "AD": "Alzheimer's Disease",
    "FTD": "Frontotemporal Dementia",
    "HC": "Healthy Control",
}

# Short clinical descriptions returned in the API response.
CLASS_DESCRIPTIONS = {
    "AD": "Alzheimer's disease is a progressive neurodegenerative disorder "
          "that gradually impairs memory, reasoning, and daily functioning.",
    "FTD": "Frontotemporal dementia primarily affects the frontal and "
           "temporal lobes, impacting behavior, personality, and language.",
    "HC": "No dementia-related pattern was detected in the analyzed EEG signal.",
}


class ModelNotLoadedError(Exception):
    """Raised when a prediction is requested before its model is loaded."""
    pass


class EEGPreprocessingError(Exception):
    """Raised when an uploaded EEG file cannot be parsed or featurized."""
    pass


# ------------------------------------------------------------------
# In-memory model registry.
# Populated once at application startup by load_all_models().
# ------------------------------------------------------------------
_models = {
    "binary": None,
    "three_class": None,
}


def load_all_models():
    """
    Loads both trained models into memory once, at application startup,
    so every /predict request reuses the same loaded model instead of
    re-loading it from disk on every call.

    >>> WHERE TO PLUG IN YOUR TRAINED MODELS <<<
    Replace the two TODO blocks below with the exact loading code from
    your training notebook. Common examples are left commented out.
    """

    # ----------------------------------------------------------------
    # TODO: LOAD YOUR BINARY CLASSIFICATION MODEL HERE
    # ----------------------------------------------------------------
    try:
        # Example (scikit-learn / joblib):
        # import joblib
        # _models["binary"] = joblib.load(BINARY_MODEL_PATH)

        # Example (TensorFlow / Keras):
        # from tensorflow.keras.models import load_model
        # _models["binary"] = load_model(BINARY_MODEL_PATH)

        # Example (PyTorch):
        # import torch
        # from my_model_definitions import BinaryEEGModel
        # _models["binary"] = BinaryEEGModel()
        # _models["binary"].load_state_dict(torch.load(BINARY_MODEL_PATH, map_location="cpu"))
        # _models["binary"].eval()

        if os.path.exists(BINARY_MODEL_PATH):
            # Placeholder — remove once real loading code above is active.
            logger.info("Binary model file found at %s (loader not yet implemented).", BINARY_MODEL_PATH)
        else:
            logger.warning("Binary model file not found at %s.", BINARY_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load binary classification model: %s", exc)
        _models["binary"] = None

    # ----------------------------------------------------------------
    # TODO: LOAD YOUR THREE-CLASS CLASSIFICATION MODEL HERE
    # ----------------------------------------------------------------
    try:
        # Example (scikit-learn / joblib):
        # import joblib
        # _models["three_class"] = joblib.load(THREE_CLASS_MODEL_PATH)

        # Example (TensorFlow / Keras):
        # from tensorflow.keras.models import load_model
        # _models["three_class"] = load_model(THREE_CLASS_MODEL_PATH)

        # Example (PyTorch):
        # import torch
        # from my_model_definitions import ThreeClassEEGModel
        # _models["three_class"] = ThreeClassEEGModel()
        # _models["three_class"].load_state_dict(torch.load(THREE_CLASS_MODEL_PATH, map_location="cpu"))
        # _models["three_class"].eval()

        if os.path.exists(THREE_CLASS_MODEL_PATH):
            # Placeholder — remove once real loading code above is active.
            logger.info("Three-class model file found at %s (loader not yet implemented).", THREE_CLASS_MODEL_PATH)
        else:
            logger.warning("Three-class model file not found at %s.", THREE_CLASS_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load three-class classification model: %s", exc)
        _models["three_class"] = None


def is_binary_model_ready():
    """Returns True once the binary model has been loaded."""
    return _models["binary"] is not None


def is_three_class_model_ready():
    """Returns True once the three-class model has been loaded."""
    return _models["three_class"] is not None


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def preprocess_eeg_file(file_path):
    """
    Loads an uploaded EEG file from disk and converts it into whatever
    input format your model expects (e.g. a feature vector, a windowed
    signal array, a spectrogram, etc.).

    >>> WHERE TO PLUG IN YOUR PREPROCESSING/FEATURE-EXTRACTION CODE <<<
    Replace the placeholder body below with the exact preprocessing
    pipeline used in your training notebook (filtering, epoching,
    feature extraction, normalization, etc.). The returned object is
    passed directly into run_binary_prediction() / run_three_class_prediction().
    """
    extension = os.path.splitext(file_path)[1].lower()

    # ----------------------------------------------------------------
    # TODO: LOAD + PREPROCESS THE RAW EEG FILE HERE
    # ----------------------------------------------------------------
    # Example for .csv / .txt (pandas):
    #   import pandas as pd
    #   raw = pd.read_csv(file_path)
    #   features = extract_features(raw)   # your notebook's feature function
    #   return features
    #
    # Example for .edf (MNE):
    #   import mne
    #   raw = mne.io.read_raw_edf(file_path, preload=True)
    #   features = extract_features_from_raw(raw)
    #   return features
    #
    # Example for .mat (SciPy):
    #   from scipy.io import loadmat
    #   mat = loadmat(file_path)
    #   features = extract_features(mat)
    #   return features
    #
    # Example for .npy (NumPy):
    #   import numpy as np
    #   signal = np.load(file_path)
    #   features = extract_features(signal)
    #   return features

    raise EEGPreprocessingError(
        f"EEG preprocessing is not yet implemented for '{extension}' files. "
        "Add your feature-extraction pipeline in preprocess_eeg_file()."
    )


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
def run_binary_prediction(file_path):
    """
    Runs the binary classification model (Alzheimer's Disease vs.
    Healthy Control) on the uploaded EEG file and returns a
    standardized result dictionary.

    Returns:
        dict: {
            "prediction": str,           # e.g. "Alzheimer's Disease"
            "confidence": float,         # e.g. 96.4
            "probabilities": dict,       # e.g. {"AD": 96.4, "HC": 3.6}
        }
    """
    if not is_binary_model_ready():
        raise ModelNotLoadedError(
            "The binary classification model is not loaded. "
            "Add your model loading code in load_all_models()."
        )

    features = preprocess_eeg_file(file_path)

    # ----------------------------------------------------------------
    # TODO: RUN YOUR BINARY MODEL INFERENCE HERE
    # ----------------------------------------------------------------
    # Example (scikit-learn):
    #   model = _models["binary"]
    #   probability_array = model.predict_proba([features])[0]  # [P(AD), P(HC)]
    #
    # Example (Keras):
    #   model = _models["binary"]
    #   probability_array = model.predict(features)[0]
    #
    # Example (PyTorch):
    #   import torch
    #   model = _models["binary"]
    #   with torch.no_grad():
    #       logits = model(features)
    #       probability_array = torch.softmax(logits, dim=-1).numpy()[0]

    raise ModelNotLoadedError(
        "Binary model inference is not yet implemented. "
        "Add your prediction code in run_binary_prediction()."
    )

    # Once inference code above is active, build the standardized
    # response the same way as below (kept here for reference):
    #
    # probabilities = {
    #     "AD": round(float(probability_array[0]) * 100, 1),
    #     "HC": round(float(probability_array[1]) * 100, 1),
    # }
    # predicted_class = max(probabilities, key=probabilities.get)
    # return {
    #     "prediction": CLASS_DISPLAY_NAMES[predicted_class],
    #     "confidence": probabilities[predicted_class],
    #     "probabilities": probabilities,
    # }


def run_three_class_prediction(file_path):
    """
    Runs the three-class classification model (Alzheimer's Disease vs.
    Frontotemporal Dementia vs. Healthy Control) on the uploaded EEG
    file and returns a standardized result dictionary.

    Returns:
        dict: {
            "prediction": str,           # e.g. "Frontotemporal Dementia"
            "confidence": float,         # e.g. 88.2
            "probabilities": dict,       # e.g. {"AD": 5.1, "FTD": 88.2, "HC": 6.7}
        }
    """
    if not is_three_class_model_ready():
        raise ModelNotLoadedError(
            "The three-class classification model is not loaded. "
            "Add your model loading code in load_all_models()."
        )

    features = preprocess_eeg_file(file_path)

    # ----------------------------------------------------------------
    # TODO: RUN YOUR THREE-CLASS MODEL INFERENCE HERE
    # ----------------------------------------------------------------
    # Example (scikit-learn):
    #   model = _models["three_class"]
    #   probability_array = model.predict_proba([features])[0]  # [P(AD), P(FTD), P(HC)]
    #
    # Example (Keras):
    #   model = _models["three_class"]
    #   probability_array = model.predict(features)[0]
    #
    # Example (PyTorch):
    #   import torch
    #   model = _models["three_class"]
    #   with torch.no_grad():
    #       logits = model(features)
    #       probability_array = torch.softmax(logits, dim=-1).numpy()[0]

    raise ModelNotLoadedError(
        "Three-class model inference is not yet implemented. "
        "Add your prediction code in run_three_class_prediction()."
    )

    # Once inference code above is active, build the standardized
    # response the same way as below (kept here for reference):
    #
    # probabilities = {
    #     "AD": round(float(probability_array[0]) * 100, 1),
    #     "FTD": round(float(probability_array[1]) * 100, 1),
    #     "HC": round(float(probability_array[2]) * 100, 1),
    # }
    # predicted_class = max(probabilities, key=probabilities.get)
    # return {
    #     "prediction": CLASS_DISPLAY_NAMES[predicted_class],
    #     "confidence": probabilities[predicted_class],
    #     "probabilities": probabilities,
    # }


def get_description_for_prediction(predicted_class_key):
    """Returns the short clinical description shown in the result card."""
    return CLASS_DESCRIPTIONS.get(
        predicted_class_key,
        "No description is available for this prediction."
    )

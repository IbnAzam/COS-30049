# backend/app/predict.py
import joblib
from pathlib import Path
from typing import Literal, Dict

# Resolve model paths relative to this file
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "Models"

email_model = joblib.load(MODELS_DIR / "email_model.pkl")
url_model   = joblib.load(MODELS_DIR / "url_model.pkl")

def classify_input(text: str) -> Dict[str, object]:
    """
    Returns:
      {
        "label": "Spam" | "Ham",
        "probability": float (0..1),
        "model_used": "email" | "url",
        "pred": int (0 or 1)
      }
    """
    # naive URL heuristic – tweak if you like
    use_url = text.strip().lower().startswith(("http://", "https://"))
    model_used: Literal["email", "url"] = "url" if use_url else "email"
    model = url_model if use_url else email_model

    pred = int(model.predict([text])[0])
    # Some models may not implement predict_proba; guard just in case
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba([text])[0][1])  # prob of class 1
    else:
        # fallback: 1.0 for class 1, 0.0 otherwise
        proba = 1.0 if pred == 1 else 0.0

    label = "Spam" if pred == 1 else "Ham"
    return {
        "label": label,
        "probability": proba,
        "model_used": model_used,
        "pred": pred
    }

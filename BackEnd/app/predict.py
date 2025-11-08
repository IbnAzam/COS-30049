# backend/app/predict.py
import joblib
from pathlib import Path
from typing import Literal, Dict, Optional
import time
from sqlmodel import Session
from .models import Prediction

# Resolve model paths relative to this file
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "Models"
EMAIL_MODEL_PATH = MODELS_DIR / "email_model.pkl"
URL_MODEL_PATH   = MODELS_DIR / "url_model.pkl"

def _safe_load(path: Path) -> Optional[object]:
    try:
        if path.exists():
            return joblib.load(path)
        else:
            print(f"⚠️  Model file not found: {path}")
            return None
    except Exception as e:
        print(f"⚠️  Failed to load model {path.name}: {e}")
        return None

email_model = _safe_load(EMAIL_MODEL_PATH)
url_model   = _safe_load(URL_MODEL_PATH)

def _predict_proba(model, text: str) -> float:
    """Return probability of class 1 (spam)."""
    if model is None:
        return 0.0
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([text])[0][1])
    pred = int(model.predict([text])[0])
    return 1.0 if pred == 1 else 0.0

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
    # Naive URL heuristic – tweak if you like
    use_url = text.strip().lower().startswith(("http://", "https://"))
    model_used: Literal["email", "url"] = "url" if use_url else "email"
    model = url_model if use_url else email_model

    if model is None:
        return {"label": "Ham", "probability": 0.0, "model_used": model_used, "pred": 0}

    pred = int(model.predict([text])[0])
    proba = _predict_proba(model, text)
    label = "Spam" if pred == 1 else "Ham"
    return {"label": label, "probability": proba, "model_used": model_used, "pred": pred}

def classify_and_log(text: str, session: Session) -> Dict[str, object]:
    """Run classify_input and persist a row to SQLite."""
    t0 = time.perf_counter()
    result = classify_input(text)

    rec = Prediction(
        text=text,
        length=len(text),
        label=result["label"],
        probability=result["probability"],
        model_used=result["model_used"],
        pred=result["pred"],
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)

    result["id"] = rec.id
    return result

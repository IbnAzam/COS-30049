# backend/app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from .schemas import PredictRequest, PredictResponse
from .db import init_db, get_session
from .predict import classify_and_log  # logging wrapper
from .stats_api import router as stats_router


app = FastAPI(title="Spam Detector API", version="1.0.0")

# Adjust origins for your dev/prod front-ends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stats_router)


@app.on_event("startup")
def _startup():
    init_db()  # create SQLite tables on boot

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, session: Session = Depends(get_session)):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    result = classify_and_log(text, session)  # logs + returns inference
    result["confidence_pct"] = result["probability"] * 100
    return PredictResponse(**result)

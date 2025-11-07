# backend/app/schemas.py
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    label: str
    probability: float  # 0..1
    model_used: str
    pred: int

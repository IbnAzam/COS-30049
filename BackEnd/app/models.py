from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field


def utcnow():
    return datetime.now(timezone.utc)  # aware UTC

class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)

    text: str
    length: int

    label: str           # "Spam" | "Ham"   
    probability: float   # 0..1
    model_used: str      # "email" | "url"
    pred: int            # 0 | 1

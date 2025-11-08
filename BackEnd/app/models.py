from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    text: str
    length: int

    label: str           # "Spam" | "Ham"
    probability: float   # 0..1
    model_used: str      # "email" | "url"
    pred: int            # 0 | 1

# backend/app/stats_api.py
from datetime import datetime, timedelta, timezone
from typing import Literal, Dict, Any, List
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select, col
from sqlalchemy import func, text
from .db import get_session, engine
from .models import Prediction

router = APIRouter(prefix="/api/stats", tags=["stats"])

# Detect dialect so we can use proper date bucketing
DIALECT = engine.url.get_backend_name()  # "sqlite", "postgresql", etc.


def _utcnow() -> datetime:
    # Keep everything in UTC internally
    return datetime.now(timezone.utc)

def to_utc_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # treat legacy naive values as UTC
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@router.get("/summary")
def summary(session: Session = Depends(get_session)) -> Dict[str, Any]:
    # Total scans
    total = session.exec(select(func.count(Prediction.id))).one()

    # By label
    by_label_rows = session.exec(
        select(Prediction.label, func.count(Prediction.id))
        .group_by(Prediction.label)
    ).all()
    by_label = [{"label": lbl or "Unknown", "c": cnt} for (lbl, cnt) in by_label_rows]

    # Average probability
    avg_prob = session.exec(select(func.avg(Prediction.probability))).one() or 0.0

    # By model
    by_model_rows = session.exec(
        select(Prediction.model_used, func.count(Prediction.id))
        .group_by(Prediction.model_used)
        .order_by(func.count(Prediction.id).desc())
    ).all()
    by_model = [{"model": m or "Unknown", "c": cnt} for (m, cnt) in by_model_rows]

    # Last 24h count (handy KPI)
    since_24h = _utcnow() - timedelta(hours=24)
    last_24h = session.exec(
        select(func.count(Prediction.id)).where(Prediction.created_at >= since_24h)
    ).one()

    return {
        "total_scans": total,
        "by_label": by_label,                    # [{label:"Spam", c:...}, ...]
        "average_probability": float(avg_prob),  # 0..1
        "by_model": by_model,                    # [{model:"email", c:...}, ...]
        "last_24h": last_24h
    }

@router.get("/latest-spam")
def latest_spam(session: Session = Depends(get_session)):
    dt = session.exec(
        select(Prediction.created_at)
        .where(Prediction.label == "Spam")
        .order_by(Prediction.created_at.desc())
        .limit(1)
    ).first()

    return {"created_at": to_utc_iso(dt)}

@router.get("/timeseries")
def timeseries(
    bucket: Literal["hour", "day", "week", "month"] = "day",
    days: int = Query(30, ge=1, le=365),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Returns points like:
      { bucket: "2025-11-08T00:00:00Z", Spam: 10, Ham: 4, Total: 14 }
    over the last N days (UTC window).
    """

    start = _utcnow() - timedelta(days=days)

    if DIALECT == "sqlite":
        # SQLite stores datetimes as TEXT by default via SQLModel; use strftime
        if bucket == "hour":
            expr = "strftime('%Y-%m-%dT%H:00:00Z', created_at)"
        elif bucket == "day":
            expr = "strftime('%Y-%m-%dT00:00:00Z', created_at)"
        elif bucket == "week":
            # ISO week label; later expand to a representative timestamp
            expr = "strftime('%Y-W%W', created_at)"
        else:  # month
            expr = "strftime('%Y-%m-01T00:00:00Z', created_at)"

        sql = text(f"""
            SELECT {expr} AS bucket, label, COUNT(*) AS c
            FROM prediction
            WHERE created_at >= :start
            GROUP BY bucket, label
            ORDER BY bucket ASC
        """)
        rows = session.exec(sql, params={"start": start.isoformat()}).all()

        # Collate buckets
        buckets = []
        spam_map, ham_map = {}, {}
        for b, label, c in rows:
            if b not in buckets:
                buckets.append(b)
            if label == "Spam":
                spam_map[b] = c
            elif label == "Ham":
                ham_map[b] = c

        out = []
        for b in buckets:
            s = spam_map.get(b, 0)
            h = ham_map.get(b, 0)
            out.append({"bucket": b, "Spam": s, "Ham": h, "Total": s + h})

        return {"bucket": bucket, "points": out}

    else:
        # Postgres etc.: use DATE_TRUNC
        if bucket == "hour":
            trunc = func.date_trunc("hour", col(Prediction.created_at))
        elif bucket == "day":
            trunc = func.date_trunc("day", col(Prediction.created_at))
        elif bucket == "week":
            trunc = func.date_trunc("week", col(Prediction.created_at))
        else:
            trunc = func.date_trunc("month", col(Prediction.created_at))

        rows = session.exec(
            select(trunc.label("bucket"), Prediction.label, func.count(Prediction.id))
            .where(Prediction.created_at >= start)
            .group_by("bucket", Prediction.label)
            .order_by("bucket")
        ).all()

        buckets = []
        spam_map, ham_map = {}, {}
        for b, label, c in rows:
            # Ensure ISO-ish string with Z for the frontend
            b_str = b.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            if b_str not in buckets:
                buckets.append(b_str)
            if label == "Spam":
                spam_map[b_str] = c
            elif label == "Ham":
                ham_map[b_str] = c

        out = []
        for b in buckets:
            s = spam_map.get(b, 0)
            h = ham_map.get(b, 0)
            out.append({"bucket": b, "Spam": s, "Ham": h, "Total": s + h})

        return {"bucket": bucket, "points": out}


@router.get("/distribution")
def distribution(
    bins: int = Query(10, ge=2, le=100),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Histogram of probability (0..1). Binned on the server.
    """
    probs: List[float] = [
        float(p) for (p,) in session.exec(select(Prediction.probability)).all()
        if p is not None
    ]
    counts = [0] * bins
    for p in probs:
        idx = min(bins - 1, max(0, int(p * bins)))
        counts[idx] += 1

    bin_edges = [i / bins for i in range(bins + 1)]
    return {"bins": bins, "counts": counts, "bin_edges": bin_edges}

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from backend.database import get_db
from backend.models import User, Signal
from backend import config

router = APIRouter()


class SignalPayload(BaseModel):
    signal: str
    confidence: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    reason: Optional[str] = ""
    timestamp: str


# ── Public endpoint: EA polls this with user's API key ──────────────────────

@router.get("/api/signal")
def get_signal(
    api_key: str,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    if not user.is_subscribed:
        raise HTTPException(status_code=403, detail="No active subscription.")

    latest = db.query(Signal).order_by(Signal.id.desc()).first()
    if not latest:
        return JSONResponse({"signal": "hold", "confidence": 0.0, "sl": None, "tp": None,
                             "reason": "No signal yet", "timestamp": datetime.now(timezone.utc).isoformat()})

    return {
        "signal": latest.signal,
        "confidence": latest.confidence,
        "sl": latest.sl,
        "tp": latest.tp,
        "reason": latest.reason,
        "timestamp": latest.timestamp,
    }


# ── Internal endpoint: signal generator posts here ──────────────────────────

@router.post("/internal/signal")
def ingest_signal(
    payload: SignalPayload,
    x_internal_secret: str = Header(...),
    db: Session = Depends(get_db),
):
    if x_internal_secret != config.INTERNAL_SIGNAL_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden.")

    record = Signal(
        signal=payload.signal,
        confidence=payload.confidence,
        sl=payload.sl,
        tp=payload.tp,
        reason=payload.reason,
        timestamp=payload.timestamp,
    )
    db.add(record)
    db.commit()
    return {"ok": True}


# ── Subscription status check (used by EA to verify before trading) ─────────

@router.get("/api/status")
def get_status(api_key: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    sub = user.active_subscription
    return {
        "subscribed": user.is_subscribed,
        "expires_at": sub.expires_at.isoformat() if sub else None,
    }

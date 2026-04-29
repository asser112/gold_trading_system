from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from backend.database import get_db
from backend.models import User, Signal, Bot
from backend import config

router = APIRouter()

# Legacy GET /api/signal and POST /internal/signal use this default bot.
DEFAULT_LEGACY_SLUG = "xgboost-v1"


class SignalPayload(BaseModel):
    signal: str
    confidence: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    reason: Optional[str] = ""
    timestamp: str


def _no_signal_json() -> JSONResponse:
    return JSONResponse(
        {
            "signal": "hold",
            "confidence": 0.0,
            "sl": None,
            "tp": None,
            "reason": "No signal yet",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def _signal_to_dict(s: Signal) -> dict:
    return {
        "signal": s.signal,
        "confidence": s.confidence,
        "sl": s.sl,
        "tp": s.tp,
        "reason": s.reason,
        "timestamp": s.timestamp,
    }


def _get_user_subscribed(db: Session, api_key: str) -> User:
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    if not user.is_subscribed:
        raise HTTPException(status_code=403, detail="No active subscription.")
    return user


# ── Public: list active bots ────────────────────────────────────────────────

@router.get("/api/bots")
def list_bots(db: Session = Depends(get_db)):
    """Public list of registered trading bots (for documentation / UI)."""
    rows = (
        db.query(Bot)
        .filter(Bot.is_active == True)  # noqa: E712
        .order_by(Bot.slug.asc())
        .all()
    )
    return [
        {"slug": b.slug, "name": b.name, "description": b.description or ""}
        for b in rows
    ]


# ── Public: legacy endpoint (defaults to xgboost-v1 stream) ───────────────
# Registered before /api/signal/{slug} so the path /api/signal is unambiguous.

@router.get("/api/signal")
def get_signal(
    api_key: str,
    db: Session = Depends(get_db),
):
    _get_user_subscribed(db, api_key)

    bot = db.query(Bot).filter(Bot.slug == DEFAULT_LEGACY_SLUG).first()
    if bot:
        latest = (
            db.query(Signal)
            .filter(Signal.bot_id == bot.id)
            .order_by(Signal.id.desc())
            .first()
        )
    else:
        latest = None

    # Fallback: no default bot yet — use latest signal globally (old behaviour)
    if not latest:
        latest = db.query(Signal).order_by(Signal.id.desc()).first()

    if not latest:
        return _no_signal_json()
    return _signal_to_dict(latest)


# ── Public: EA polls by bot slug ────────────────────────────────────────────

@router.get("/api/signal/{slug}")
def get_signal_by_slug(
    slug: str,
    api_key: str,
    db: Session = Depends(get_db),
):
    _get_user_subscribed(db, api_key)
    bot = db.query(Bot).filter(Bot.slug == slug, Bot.is_active == True).first()  # noqa: E712
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown or inactive bot slug.")

    latest = (
        db.query(Signal)
        .filter(Signal.bot_id == bot.id)
        .order_by(Signal.id.desc())
        .first()
    )
    if not latest:
        return _no_signal_json()
    return _signal_to_dict(latest)


# ── Internal: ingest by slug ────────────────────────────────────────────────

@router.post("/internal/signal/{slug}")
def ingest_signal_by_slug(
    slug: str,
    payload: SignalPayload,
    x_internal_secret: str = Header(...),
    db: Session = Depends(get_db),
):
    if x_internal_secret != config.INTERNAL_SIGNAL_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden.")

    bot = db.query(Bot).filter(Bot.slug == slug).first()
    if not bot:
        raise HTTPException(status_code=404, detail=f"Unknown bot slug: {slug}")

    record = Signal(
        bot_id=bot.id,
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


# ── Internal: legacy ingest (defaults to xgboost-v1) ────────────────────────

@router.post("/internal/signal")
def ingest_signal(
    payload: SignalPayload,
    x_internal_secret: str = Header(...),
    db: Session = Depends(get_db),
):
    if x_internal_secret != config.INTERNAL_SIGNAL_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden.")

    bot = db.query(Bot).filter(Bot.slug == DEFAULT_LEGACY_SLUG).first()
    record = Signal(
        bot_id=bot.id if bot else None,
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


# ── Subscription status check (used by EA) ────────────────────────────────

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

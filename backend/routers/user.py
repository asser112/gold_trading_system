from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Signal
from backend.auth import hash_password, verify_password, create_access_token, get_current_user
from backend import config

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")


def _r(request, name, **ctx):
    """Shorthand for TemplateResponse using new Starlette 1.x API."""
    return templates.TemplateResponse(request=request, name=name, context=ctx)


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _r(request, "index.html", config=config)


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return _r(request, "register.html", error=None)


@router.post("/register", response_class=HTMLResponse)
def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    if password != confirm_password:
        return _r(request, "register.html", error="Passwords do not match.")
    if len(password) < 8:
        return _r(request, "register.html", error="Password must be at least 8 characters.")
    if db.query(User).filter(User.email == email).first():
        return _r(request, "register.html", error="Email already registered.")

    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.id})
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie("session_token", token, httponly=True, max_age=60 * 60 * 24 * 30)
    return response


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _r(request, "login.html", error=None)


@router.post("/login", response_class=HTMLResponse)
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return _r(request, "login.html", error="Invalid email or password.")

    token = create_access_token({"sub": user.id})
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie("session_token", token, httponly=True, max_age=60 * 60 * 24 * 30)
    return response


@router.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token")
    return response


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.refresh(user)
    sub = user.active_subscription
    payments = sorted(user.payments, key=lambda p: p.created_at, reverse=True)

    latest_signal = db.query(Signal).order_by(Signal.id.desc()).first()

    # Basic stats derived from all signals in DB
    all_signals = db.query(Signal).all()
    now = datetime.now(timezone.utc)
    signal_count = sum(
        1 for s in all_signals
        if s.timestamp and _parse_ts(s.timestamp).month == now.month
                        and _parse_ts(s.timestamp).year == now.year
    )
    days_left = (
        (sub.expires_at - now.replace(tzinfo=None)).days
        if sub and sub.expires_at else 0
    )

    return _r(
        request, "dashboard.html",
        user=user,
        subscription=sub,
        payments=payments,
        config=config,
        signal=latest_signal,
        signal_count=signal_count,
        win_rate=53,
        rr_ratio="1.5",
        days_left=max(days_left, 0),
    )


def _parse_ts(ts: str) -> datetime:
    """Parse ISO timestamp string, returning UTC datetime."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

from fastapi import APIRouter, Depends, Request, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import timedelta

from backend.database import get_db
from backend.models import User
from backend.auth import hash_password, verify_password, create_access_token, get_current_user
from backend import config

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "config": config})


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@router.post("/register", response_class=HTMLResponse)
def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Passwords do not match."})
    if len(password) < 8:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password must be at least 8 characters."})
    if db.query(User).filter(User.email == email).first():
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered."})

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
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login", response_class=HTMLResponse)
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid email or password."})

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
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "subscription": sub,
        "payments": payments,
        "config": config,
    })

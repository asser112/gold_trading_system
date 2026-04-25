import hashlib
import hmac
import json
import httpx
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Payment, Subscription
from backend.auth import get_current_user
from backend import config

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")

NOWPAYMENTS_HEADERS = {
    "x-api-key": config.NOWPAYMENTS_API_KEY,
    "Content-Type": "application/json",
}


@router.get("/pay", response_class=HTMLResponse)
def pay_page(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("pay.html", {
        "request": request,
        "user": user,
        "coins": config.COIN_LABELS,
        "price": config.SUBSCRIPTION_PRICE_USD,
    })


@router.post("/pay/create")
async def create_payment(
    request: Request,
    coin: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if coin not in config.ACCEPTED_COINS:
        raise HTTPException(status_code=400, detail="Unsupported coin.")

    payload = {
        "price_amount": config.SUBSCRIPTION_PRICE_USD,
        "price_currency": "usd",
        "pay_currency": coin,
        "order_id": f"{user.id}-{int(datetime.now().timestamp())}",
        "order_description": f"Gold Trading Signal — {config.SUBSCRIPTION_DAYS}-day subscription",
        "ipn_callback_url": f"{config.BASE_URL}/webhooks/nowpayments",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{config.NOWPAYMENTS_API_URL}/payment",
            headers=NOWPAYMENTS_HEADERS,
            json=payload,
            timeout=15,
        )

    if resp.status_code != 201:
        raise HTTPException(status_code=502, detail="Payment gateway error. Try again.")

    data = resp.json()

    payment = Payment(
        user_id=user.id,
        nowpayments_id=str(data["payment_id"]),
        pay_currency=data["pay_currency"],
        pay_amount=float(data["pay_amount"]),
        price_usd=config.SUBSCRIPTION_PRICE_USD,
        pay_address=data.get("pay_address"),
        status="waiting",
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)

    return templates.TemplateResponse("payment_pending.html", {
        "request": request,
        "payment": payment,
        "coin_label": config.COIN_LABELS.get(coin, coin),
    })


@router.post("/webhooks/nowpayments")
async def nowpayments_webhook(request: Request, db: Session = Depends(get_db)):
    body = await request.body()

    # Verify HMAC-SHA512 signature
    sig = request.headers.get("x-nowpayments-sig", "")
    expected = hmac.new(
        config.NOWPAYMENTS_IPN_SECRET.encode(),
        body,
        hashlib.sha512,
    ).hexdigest()

    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=400, detail="Invalid signature.")

    data = json.loads(body)
    nowpayments_id = str(data.get("payment_id", ""))
    new_status = data.get("payment_status", "")

    payment = db.query(Payment).filter(Payment.nowpayments_id == nowpayments_id).first()
    if not payment:
        return {"ok": True}  # unknown payment, ignore

    payment.status = new_status

    if new_status in ("finished", "confirmed"):
        payment.confirmed_at = datetime.now(timezone.utc)

        # Extend or create subscription
        now = datetime.now(timezone.utc)
        user = db.query(User).filter(User.id == payment.user_id).first()
        existing = user.active_subscription

        if existing and existing.expires_at > now:
            existing.expires_at += timedelta(days=config.SUBSCRIPTION_DAYS)
        else:
            sub = Subscription(
                user_id=user.id,
                status="active",
                expires_at=now + timedelta(days=config.SUBSCRIPTION_DAYS),
            )
            db.add(sub)

    db.commit()
    return {"ok": True}

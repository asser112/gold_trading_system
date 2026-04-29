"""
Create an admin/test user with a known API key and active subscription.
Also seeds default bots in the multi-bot registry.

Run from project root:
    python backend/create_admin.py
"""
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import engine, Base, SessionLocal
from backend.models import User, Subscription, Bot
from backend.auth import hash_password
from backend.bot_defaults import seed_default_bots

EMAIL    = "admin@goldsignal.local"
PASSWORD = "admin1234"
API_KEY  = "admin-test-key-00000000"   # fixed key — easy to paste into EA


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    seed_default_bots(db)

    existing = db.query(User).filter(User.email == EMAIL).first()
    if existing:
        print(f"User already exists: {existing.email}")
        print(f"API key : {existing.api_key}")
        sub = existing.active_subscription
        if sub:
            print(f"Sub expires: {sub.expires_at.date()}")
        else:
            # Add subscription if missing
            sub = Subscription(
                user_id=existing.id,
                status="active",
                expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            )
            db.add(sub)
            db.commit()
            print("Subscription added (1 year).")
        db.close()
        return

    user = User(
        email=EMAIL,
        password_hash=hash_password(PASSWORD),
        api_key=API_KEY,
    )
    db.add(user)
    db.flush()

    sub = Subscription(
        user_id=user.id,
        status="active",
        expires_at=datetime.now(timezone.utc) + timedelta(days=365),
    )
    db.add(sub)
    db.commit()
    db.close()

    print("=" * 50)
    print("Admin user created")
    print("=" * 50)
    print(f"Email   : {EMAIL}")
    print(f"Password: {PASSWORD}")
    print(f"API key : {API_KEY}")
    print(f"Sub     : active for 365 days")
    print("=" * 50)
    print("Use this API key in MT5 EA inputs or for testing:")
    print("  GET /api/signal (legacy, xgboost-v1 stream)")
    print("  GET /api/signal/xgboost-v1")
    print("  GET /api/signal/lgbm-session-v1")

if __name__ == "__main__":
    main()

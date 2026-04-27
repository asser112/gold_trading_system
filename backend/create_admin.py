"""
Create an admin/test user with a known API key and active subscription.
Run from project root:
    python backend/create_admin.py
"""
import sys
import os
import uuid
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import engine, Base, SessionLocal
from backend.models import User, Subscription
from backend.auth import hash_password

EMAIL    = "admin@goldsignal.local"
PASSWORD = "admin1234"
API_KEY  = "admin-test-key-00000000"   # fixed key — easy to paste into EA

def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

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
    print("Use this API key in MT5 EA inputs or for testing the /api/signal endpoint.")

if __name__ == "__main__":
    main()

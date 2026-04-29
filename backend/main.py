from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from backend.database import engine, Base, SessionLocal
from backend.models import (  # noqa: F401 — ensure tables are registered
    User,
    Subscription,
    Payment,
    Signal,
    Bot,
)
from backend.routers import user, payments, signals
from backend.bot_defaults import seed_default_bots

app = FastAPI(title="Gold Trading Signal Service", docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory="backend/static"), name="static")

app.include_router(user.router)
app.include_router(payments.router)
app.include_router(signals.router)


def _migrate_add_bot_id_column() -> None:
    """Add bot_id to existing SQLite signals table (idempotent)."""
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER TABLE signals ADD COLUMN bot_id INTEGER"))
        except Exception as e:
            err = str(e).lower()
            if "duplicate column" in err or "already exists" in err:
                return
            raise


@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)
    _migrate_add_bot_id_column()
    db = SessionLocal()
    try:
        seed_default_bots(db)
    finally:
        db.close()

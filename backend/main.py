from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect

from backend.database import engine, Base
from backend.models import User, Subscription, Payment, Signal  # noqa: F401 — ensure tables are registered
from backend.routers import user, payments, signals

app = FastAPI(title="Gold Trading Signal Service", docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory="backend/static"), name="static")

app.include_router(user.router)
app.include_router(payments.router)
app.include_router(signals.router)


@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)

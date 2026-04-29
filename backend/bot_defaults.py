"""Default bots for the multi-bot registry (shared by main.py and create_admin.py)."""

from sqlalchemy.orm import Session

from backend.models import Bot

DEFAULT_BOTS = [
    {
        "slug": "xgboost-v1",
        "name": "XGBoost v1",
        "description": "XGBoost with SMC features",
    },
    {
        "slug": "lgbm-session-v1",
        "name": "LightGBM Session v1",
        "description": "LightGBM trained on London+NY sessions only",
    },
]


def seed_default_bots(db: Session) -> None:
    """Insert default bots if missing (idempotent)."""
    for row in DEFAULT_BOTS:
        if db.query(Bot).filter(Bot.slug == row["slug"]).first():
            continue
        db.add(
            Bot(
                slug=row["slug"],
                name=row["name"],
                description=row["description"],
                is_active=True,
            )
        )
    db.commit()

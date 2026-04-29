import os
from dotenv import load_dotenv

# Support running a second instance with a different .env file.
# Set ENV_FILE env var before starting uvicorn, e.g.:
#   set ENV_FILE=backend/.env.lgbm && python -m uvicorn backend.main:app --port 8001
_env_file = os.environ.get("ENV_FILE")
load_dotenv(dotenv_path=_env_file)  # dotenv_path=None falls back to auto-discovery (.env)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./backend/trading_saas.db")
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30 days

NOWPAYMENTS_API_KEY = os.getenv("NOWPAYMENTS_API_KEY", "")
NOWPAYMENTS_IPN_SECRET = os.getenv("NOWPAYMENTS_IPN_SECRET", "")
NOWPAYMENTS_API_URL = "https://api.nowpayments.io/v1"

INTERNAL_SIGNAL_SECRET = os.getenv("INTERNAL_SIGNAL_SECRET", "change-this-internal-secret")

SUBSCRIPTION_PRICE_USD = float(os.getenv("SUBSCRIPTION_PRICE_USD", "50.0"))
SUBSCRIPTION_DAYS = int(os.getenv("SUBSCRIPTION_DAYS", "30"))

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

ACCEPTED_COINS = ["usdterc20", "btc", "eth"]
COIN_LABELS = {
    "usdterc20": "USDT (ERC-20)",
    "btc": "Bitcoin (BTC)",
    "eth": "Ethereum (ETH)",
}

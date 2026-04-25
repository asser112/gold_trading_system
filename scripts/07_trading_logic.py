"""
07_trading_logic.py - Signal Generator for Gold Trading EA

Generates trading signals and writes them to signal.txt for the MT5 EA to read.
Uses the trained ML model to predict signals on live data.

Usage:
    python 07_trading_logic.py
"""

import sys
import os
import json
import logging
import time
import yaml
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "config.yaml", "r") as _f:
    _config = yaml.safe_load(_f)

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "signal_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SIGNAL_FILE = PROJECT_ROOT / "mt5_ea" / "signal.txt"
SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)

# Mirror signal into the MT5 terminal Files folder so the EA can read it directly.
# Set trading.mt5_terminal_id in config.yaml to your terminal's data folder ID.
# Find it at: Help > About in MT5, or check %APPDATA%\MetaQuotes\Terminal\
_terminal_id = _config.get("trading", {}).get("mt5_terminal_id", "")
_appdata = os.environ.get("APPDATA", "")
MT5_SIGNAL_FILE: Path | None = None
if _appdata and _terminal_id and _terminal_id != "YOUR_TERMINAL_ID":
    MT5_FILES = Path(_appdata) / "MetaQuotes" / "Terminal" / _terminal_id / "MQL5" / "Files"
    try:
        MT5_FILES.mkdir(parents=True, exist_ok=True)
        MT5_SIGNAL_FILE = MT5_FILES / "signal.txt"
    except OSError:
        logger.warning("Could not create MT5 Files directory. Check mt5_terminal_id in config.yaml.")
        MT5_SIGNAL_FILE = None
else:
    if _terminal_id == "YOUR_TERMINAL_ID":
        logger.warning("mt5_terminal_id not set in config.yaml — signal will only write to mt5_ea/signal.txt")


def write_signal(signal, confidence=0.0, sl=None, tp=None, reason=""):
    """Write a signal to the JSON file for the MT5 EA."""
    data = {
        "signal": signal.lower(),
        "confidence": round(confidence, 4),
        "sl": sl,
        "tp": tp,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    }
    content = json.dumps(data, indent=2)
    try:
        SIGNAL_FILE.write_text(content, encoding="utf-8")
        if MT5_SIGNAL_FILE is not None:
            MT5_SIGNAL_FILE.write_text(content, encoding="utf-8")
        logger.info(f"Signal: {signal.upper()} (conf={confidence:.2f}) -> {SIGNAL_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to write signal: {e}")
        return False


def get_model_prediction():
    """Get prediction from the trained model."""
    try:
        from src.trading_bot import TradingBot
        bot = TradingBot()
        bot.initialize(mode="backtest")
        logger.info("Loading trained model...")
        try:
            bot.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"No trained model found: {e}")
            bot.model = None
        return bot
    except Exception as e:
        logger.error(f"Failed to initialize trading bot: {e}")
        return None


def get_technical_signal():
    """Generate signal from technical indicators as fallback."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return "hold", 0.0, None, None
        symbol = "XAUUSDr"
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) < 50:
            mt5.shutdown()
            return "hold", 0.0, None, None
        closes = rates['close']
        ema_fast = calculate_ema(closes, 8)
        ema_slow = calculate_ema(closes, 21)
        ema_trend = calculate_ema(closes, 50)
        rsi = calculate_rsi(closes, 14)
        current_price = closes[-1]
        signal = "hold"
        confidence = 0.0
        if ema_fast > ema_slow and current_price > ema_trend and 30 < rsi < 70:
            signal = "buy"
            confidence = min(0.9, 0.5 + (ema_fast - ema_slow) / abs(ema_slow) * 100)
        elif ema_fast < ema_slow and current_price < ema_trend and 30 < rsi < 70:
            signal = "sell"
            confidence = min(0.9, 0.5 + (ema_slow - ema_fast) / abs(ema_slow) * 100)
        sl = tp = None
        if signal != "hold":
            atr = calculate_atr(rates, 14)
            if signal == "buy":
                sl = round(current_price - atr * 2, 2)
                tp = round(current_price + atr * 3, 2)
            else:
                sl = round(current_price + atr * 2, 2)
                tp = round(current_price - atr * 3, 2)
        mt5.shutdown()
        return signal, confidence, sl, tp
    except ImportError:
        logger.warning("MetaTrader5 not available, using hold signal")
        return "hold", 0.0, None, None
    except Exception as e:
        logger.error(f"Technical signal error: {e}")
        return "hold", 0.0, None, None


def calculate_ema(data, period):
    """Calculate Exponential Moving Average."""
    if len(data) < period:
        return data[-1] if len(data) > 0 else 0
    multiplier = 2 / (period + 1)
    ema = data[:period].mean()
    for price in data[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index."""
    if len(data) < period + 1:
        return 50
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(rates, period=14):
    """Calculate Average True Range."""
    if len(rates) < period + 1:
        return 5.0
    true_ranges = []
    for i in range(1, len(rates)):
        high = rates[i]['high']
        low = rates[i]['low']
        prev_close = rates[i-1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / period


def main():
    """Main signal generation loop."""
    print("=" * 60)
    print("Gold Trading System - Signal Generator")
    print("=" * 60)
    print(f"Signal file: {SIGNAL_FILE}")
    print(f"Symbol: XAUUSDr")
    print(f"Timeframe: M5")
    print("Press Ctrl+C to stop\n")

    bot = get_model_prediction()
    interval = 60
    iteration = 0
    last_signal = ""

    try:
        while True:
            iteration += 1
            logger.info(f"\n--- Cycle {iteration} ---")

            signal = "hold"
            confidence = 0.0
            sl = tp = None
            reason = ""

            if bot and bot.model is not None:
                try:
                    df = bot.data_loader.get_sample_data(100)
                    df = bot.feature_engineer.create_features(df)
                    signals = bot.model.predict(df)
                    pred = signals[-1]
                    if pred == 1:
                        signal = "buy"
                        confidence = 0.75
                        reason = "ML model buy signal"
                    elif pred == -1:
                        signal = "sell"
                        confidence = 0.75
                        reason = "ML model sell signal"
                    else:
                        reason = "ML model no clear signal"
                except Exception as e:
                    logger.error(f"ML prediction error: {e}")
                    signal, confidence, sl, tp = get_technical_signal()
                    reason = f"Technical fallback ({signal})"
            else:
                signal, confidence, sl, tp = get_technical_signal()
                reason = f"Technical indicator ({signal})"

            write_signal(signal, confidence, sl, tp, reason)
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        write_signal("hold", 0.0, reason="Signal generator stopped")
        if bot:
            bot.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()

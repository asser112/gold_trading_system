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
import urllib.request
import urllib.error
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


_backend_url = _config.get("backend", {}).get("url", "")
_internal_secret = _config.get("backend", {}).get("internal_signal_secret", "")


def _push_signal_to_backend(data: dict) -> None:
    """POST signal to the backend API so subscribers' EAs can fetch it."""
    if not _backend_url or not _internal_secret:
        return
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{_backend_url}/internal/signal",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-internal-secret": _internal_secret,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                logger.warning(f"Backend signal push returned HTTP {resp.status}")
    except urllib.error.URLError as e:
        logger.warning(f"Backend signal push failed: {e}")


def write_signal(signal, confidence=0.0, sl=None, tp=None, reason=""):
    """Write signal to local file and push to backend API."""
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
        _push_signal_to_backend(data)
        return True
    except Exception as e:
        logger.error(f"Failed to write signal: {e}")
        return False


def get_xgboost_signal(rates):
    """Generate signal using the trained XGBoost model and live MT5 data."""
    try:
        import joblib
        import numpy as np

        model_path = PROJECT_ROOT / "models" / "xgboost" / "xgboost_best.pkl"
        if not model_path.exists():
            return None, None, None, None

        model = joblib.load(model_path)
        closes = rates['close']
        highs  = rates['high']
        lows   = rates['low']

        # Compute features that match training
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        rsi   = calculate_rsi(closes, 14)
        atr   = calculate_atr(rates, 14)

        # Bollinger Bands (20-period)
        window = closes[-20:]
        bb_middle = float(np.mean(window))
        bb_std    = float(np.std(window))
        bb_upper  = bb_middle + 2 * bb_std
        bb_lower  = bb_middle - 2 * bb_std
        bb_width  = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0

        # VWAP approximation (last 20 bars)
        vwap = float(np.mean((rates['high'][-20:] + rates['low'][-20:] + rates['close'][-20:]) / 3))

        # ADX approximation using ATR ratio
        adx = min(100.0, atr / closes[-1] * 10000) if closes[-1] != 0 else 25.0

        now = datetime.utcnow()
        hour        = now.hour
        day_of_week = now.weekday()
        session_Asian  = 1 if 0  <= hour < 8  else 0
        session_London = 1 if 8  <= hour < 16 else 0
        session_NY     = 1 if 13 <= hour < 22 else 0

        # Build feature vector using model's own feature order
        feature_map = {
            'rsi': rsi, 'ema20': ema20, 'ema50': ema50, 'vwap': vwap,
            'bb_upper': bb_upper, 'bb_middle': bb_middle, 'bb_lower': bb_lower,
            'bb_width': bb_width, 'adx': adx,
            'order_block': 0.0, 'fvg_distance': 0.0,
            'liquidity_zone': 0.0, 'sweep': 0.0, 'sentiment_score': 0.0,
            'hour': hour, 'day_of_week': day_of_week,
            'session_Asian': session_Asian, 'session_London': session_London,
            'session_NY': session_NY,
            # some model versions include atr
            'atr': atr,
        }

        if hasattr(model, 'feature_names_in_'):
            feature_vector = np.array([[feature_map.get(f, 0.0) for f in model.feature_names_in_]], dtype=np.float32)
        else:
            feature_vector = np.array([[feature_map[f] for f in [
                'rsi','ema20','ema50','vwap','bb_upper','bb_middle','bb_lower',
                'bb_width','adx','order_block','fvg_distance','liquidity_zone',
                'sweep','sentiment_score','hour','day_of_week',
                'session_Asian','session_London','session_NY',
            ]]], dtype=np.float32)

        probs          = model.predict_proba(feature_vector)[0]
        predicted_class = int(np.argmax(probs))
        confidence      = float(probs[predicted_class])
        threshold       = _config.get("models", {}).get("ensemble", {}).get("confidence_threshold", 0.80)

        if confidence < threshold:
            return "hold", confidence, None, None

        current_price = float(closes[-1])
        if predicted_class == 2:      # buy
            sl = round(current_price - atr * 2, 2)
            tp = round(current_price + atr * 3, 2)
            return "buy", confidence, sl, tp
        elif predicted_class == 0:    # sell
            sl = round(current_price + atr * 2, 2)
            tp = round(current_price - atr * 3, 2)
            return "sell", confidence, sl, tp
        else:
            return "hold", confidence, None, None

    except Exception as e:
        logger.warning(f"XGBoost prediction error: {e}")
        return None, None, None, None


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

    interval = 60
    iteration = 0

    try:
        while True:
            iteration += 1
            logger.info(f"\n--- Cycle {iteration} ---")

            signal = "hold"
            confidence = 0.0
            sl = tp = None
            reason = ""

            # Try XGBoost first using live MT5 rates
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    symbol = _config.get("trading", {}).get("symbol", "XAUUSD")
                    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
                    mt5.shutdown()
                    if rates is not None and len(rates) >= 50:
                        xgb_signal, xgb_conf, xgb_sl, xgb_tp = get_xgboost_signal(rates)
                        if xgb_signal is not None:
                            signal, confidence, sl, tp = xgb_signal, xgb_conf, xgb_sl, xgb_tp
                            reason = f"XGBoost ({signal}, conf={confidence:.2f})"
                            logger.info(f"XGBoost signal: {signal} | conf={confidence:.2f} | SL={sl} | TP={tp}")
                        else:
                            signal, confidence, sl, tp = get_technical_signal()
                            reason = f"Technical fallback ({signal})"
                    else:
                        signal, confidence, sl, tp = get_technical_signal()
                        reason = "Technical fallback (no MT5 data)"
                else:
                    signal, confidence, sl, tp = get_technical_signal()
                    reason = "Technical fallback (MT5 init failed)"
            except ImportError:
                signal, confidence, sl, tp = get_technical_signal()
                reason = f"Technical indicator ({signal})"

            write_signal(signal, confidence, sl, tp, reason)
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        write_signal("hold", 0.0, reason="Signal generator stopped")
        logger.info("Done")


if __name__ == "__main__":
    main()

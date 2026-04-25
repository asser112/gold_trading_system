---
name: mt5-signal-integration
description: Guides debugging and working with the MT5 EA signal integration for the gold trading system. Use when working with signal.txt, the MT5 Expert Advisor, signal generation logic, or troubleshooting why the EA isn't trading.
---

# MT5 Signal Integration

## Signal contract

`mt5_ea/signal.txt` is the interface between Python and MT5. The EA polls this file every `poll_interval_seconds` (default: 30s).

```json
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.6478,
  "sl": 4623.54,
  "tp": 4556.06,
  "reason": "Technical indicator (sell)",
  "timestamp": "2026-04-02T16:37:38.642361"
}
```

Rules enforced by EA:
- `confidence` must exceed `models.ensemble.confidence_threshold` (0.60) — else treated as `hold`
- `signal` is case-insensitive (lowercased before write)
- Stale timestamp > `poll_interval_seconds` → EA skips

## File write locations

`scripts/07_trading_logic.py` writes to **two** locations simultaneously:
1. `mt5_ea/signal.txt` (repo copy)
2. `%APPDATA%\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\signal.txt` (MT5 sandbox)

On macOS/Linux the APPDATA path silently fails — only the repo copy is written. This is fine for development; only matters on Windows with MT5 running.

## Signal generator

```bash
# Start live signal loop (polls every 30s)
python scripts/07_trading_logic.py

# One-shot manual signal write (for testing)
python -c "
import json; from pathlib import Path; from datetime import datetime
Path('mt5_ea/signal.txt').write_text(json.dumps({
  'signal': 'buy', 'confidence': 0.75, 'sl': 3200.0, 'tp': 3350.0,
  'reason': 'manual test', 'timestamp': datetime.now().isoformat()
}, indent=2))
print('Signal written.')
"
```

## EA files

- `mt5_ea/gold_trading_ea.mq5` — main EA (reads signal file, manages orders/risk)
- `mt5_ea/signal_reader.mqh` — helper include for JSON parsing in MQL5

## Config keys that affect EA behaviour

```yaml
trading:
  signal_file: ...     # must match where Python writes
  lot_size: 0.03       # fixed lot (if risk_percent: 0)
  risk_percent: 0      # set > 0 to use % risk sizing instead
  max_spread: 30       # in points; EA skips entry if spread exceeds this
  magic_number: 123456
  atr_multiplier_sl: 1.5
  atr_multiplier_tp: 2.5
```

## Troubleshooting

| Symptom | Check |
|---------|-------|
| EA not opening trades | `confidence` < threshold; `signal` == `"hold"`; spread > `max_spread` |
| Signal file not updating | Is `07_trading_logic.py` running? Check `logs/signal_generator.log` |
| Stale timestamp | Generator crashed; restart it |
| Wrong SL/TP | ATR calculation in `07_trading_logic.py`; adjust `atr_multiplier_sl/tp` |
| EA errors in MT5 journal | Check MQL5 Experts log; common: wrong file path in EA init |

## Checking current signal

```bash
cat mt5_ea/signal.txt
python -c "import json; d=json.load(open('mt5_ea/signal.txt')); print(d['signal'], d['confidence'])"
```

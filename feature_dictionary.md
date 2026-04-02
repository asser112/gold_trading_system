# Feature Dictionary

## Market Features
- **rsi**: Relative Strength Index (14 periods)
- **atr**: Average True Range (14 periods)
- **ema20**: Exponential Moving Average (20 periods)
- **ema50**: Exponential Moving Average (50 periods)
- **vwap**: Volume Weighted Average Price (cumulative)
- **bb_upper**: Bollinger Bands Upper (20,2)
- **bb_middle**: Bollinger Bands Middle (20 SMA)
- **bb_lower**: Bollinger Bands Lower (20,2)
- **bb_width**: (bb_upper - bb_lower) / bb_middle
- **adx**: Average Directional Index (14 periods)

## Smart Money Concepts
- **order_block**: Binary indicator (1 if current price is near a previously identified order block)
- **fvg_distance**: Distance to the nearest Fair Value Gap (in price units)
- **liquidity_zone**: Binary indicator (1 if price is near a high-volume node)
- **sweep**: Binary indicator (1 if price has recently swept a previous high/low and reversed)

## News Features
- **sentiment_score**: Average FinBERT sentiment score over the candle interval (range -1 to 1)
- **economic_impact**: Surprise score from economic releases (if available)

## Time Features
- **hour**: Hour of day (0-23 UTC)
- **day_of_week**: Day of week (Monday=0 to Sunday=6)
- **session_Asian**, **session_London**, **session_NY**: One-hot encoded trading sessions
import sqlite3
import pandas as pd
import numpy as np
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def get_db_connection(db_path='data/gold_trading.db'):
    """Return a SQLite connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def retry(max_attempts=3, delay=1, backoff=2):
    """Decorator to retry a function on exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.warning(f"Attempt {attempts} failed: {e}")
                    if attempts == max_attempts:
                        raise
                    time.sleep(delay * (backoff ** (attempts - 1)))
            return None
        return wrapper
    return decorator

def load_ohlc(interval='m5', start=None, end=None, conn=None):
    """Load OHLC data from database."""
    if conn is None:
        conn = get_db_connection()
    query = f"SELECT * FROM ohlc_{interval} ORDER BY timestamp"
    if start:
        query += f" WHERE timestamp >= '{start}'"
    if end:
        query += f" AND timestamp <= '{end}'"
    df = pd.read_sql(query, conn, index_col='timestamp', parse_dates=['timestamp'])
    return df

def save_to_db(df, table_name, conn=None, if_exists='append'):
    """Save DataFrame to SQLite."""
    if conn is None:
        conn = get_db_connection()
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)

def compute_atr(df, period=14):
    """Compute ATR from OHLC."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr
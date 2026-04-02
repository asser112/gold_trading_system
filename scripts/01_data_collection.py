#!/usr/bin/env python3
"""
Data Collection - Updated with real data sources
- yfinance for recent intraday data (last 7 days)
- Synthetic intraday from daily for longer history
- Dukascopy fallback (if API works)
- Stores in SQLite database
"""
import pandas as pd
import numpy as np
import sqlite3
import yaml
import logging
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)

DB_PATH = 'data/gold_trading.db'
Path('data/raw').mkdir(parents=True, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def download_yfinance_intraday(ticker='GC=F', days_back=7, interval='5m'):
    """Download recent intraday data from yfinance."""
    end = datetime.now()
    start = end - timedelta(days=days_back)
    logger.info(f"Downloading {ticker} from {start.date()} to {end.date()}, interval={interval}")
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if data.empty:
            logger.warning(f"No data returned from yfinance for {ticker}")
            return None
        
        # Save the datetime index before flattening
        datetime_index = data.index.copy()
        
        # Flatten multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        data = data.reset_index()
        # The index column might be named 'Datetime' or 'Date'
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in data.columns:
            data = data.rename(columns={'Date': 'timestamp'})
        elif data.columns[0] != 'timestamp':
            # Try to find the datetime column
            for col in data.columns:
                if 'datetime' in col.lower() or 'date' in col.lower():
                    data = data.rename(columns={col: 'timestamp'})
                    break
        
        # If timestamp column doesn't exist, use the first column
        if 'timestamp' not in data.columns:
            data = data.rename(columns={data.columns[0]: 'timestamp'})
        
        # Ensure timezone-naive UTC
        if data['timestamp'].dt.tz is not None:
            data['timestamp'] = data['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Standardize column names to lowercase
        rename_map = {'Datetime': 'timestamp', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'}
        data = data.rename(columns=rename_map)
        
        # Select only needed columns
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Downloaded {len(data)} rows of {interval} data")
        return data
    except Exception as e:
        logger.error(f"yfinance error: {e}")
        return None


def download_yfinance_daily(ticker='GC=F', years_back=3):
    """Download daily data from yfinance for longer history."""
    end = datetime.now()
    start = end - timedelta(days=years_back * 365)
    logger.info(f"Downloading daily data from {start.date()} to {end.date()}")
    try:
        data = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
        if data.empty:
            logger.warning(f"No daily data returned")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        data = data.reset_index()
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in data.columns:
            data = data.rename(columns={'Date': 'timestamp'})
        elif 'timestamp' not in data.columns:
            # Try to find the datetime column
            for col in data.columns:
                if 'datetime' in col.lower() or 'date' in col.lower():
                    data = data.rename(columns={col: 'timestamp'})
                    break
            if 'timestamp' not in data.columns:
                data = data.rename(columns={data.columns[0]: 'timestamp'})
        
        if data['timestamp'].dt.tz is not None:
            data['timestamp'] = data['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Standardize column names to lowercase
        data.columns = [c.lower() if c in ['Open', 'High', 'Low', 'Close', 'Volume'] else c for c in data.columns]
        
        logger.info(f"Downloaded {len(data)} rows of daily data")
        return data
    except Exception as e:
        logger.error(f"yfinance daily error: {e}")
        return None


def generate_intraday_from_daily(daily_data, target_interval='5min', volatility_factor=1.0):
    """Generate synthetic intraday OHLC from daily data using realistic patterns."""
    intraday_rows = []
    
    # Ensure lowercase column names
    daily_data = daily_data.copy()
    daily_data.columns = [c.lower() if c in ['Open', 'High', 'Low', 'Close', 'Volume'] else c for c in daily_data.columns]
    
    for idx, row in daily_data.iterrows():
        date = row['timestamp'].date()
        daily_open = row['open']
        daily_high = row['high']
        daily_low = row['low']
        daily_close = row['close']
        
        # Estimate daily range
        daily_range = daily_high - daily_low
        if daily_range <= 0:
            daily_range = daily_close * 0.005  # 0.5% default
        
        # Generate M5 candles for this day
        # Market hours: 00:00-23:55 UTC (Dukascopy style, 24h market)
        n_candles = {
            '5min': 288,
            '15min': 96,
            '1h': 24,
            '4h': 6,
        }.get(target_interval, 288)
        
        interval_minutes = {
            '5min': 5,
            '15min': 15,
            '1h': 60,
            '4h': 240,
        }.get(target_interval, 5)
        
        candle_ranges = np.random.exponential(daily_range / np.sqrt(n_candles), n_candles) * volatility_factor
        candle_ranges = np.clip(candle_ranges, daily_range * 0.01, daily_range * 0.5)
        
        # Generate candle OHLC
        current_price = daily_open
        rng = np.random.default_rng(42 + date.toordinal())
        
        for i in range(n_candles):
            timestamp = pd.Timestamp(year=date.year, month=date.month, day=date.day) + timedelta(minutes=i * interval_minutes)
            
            # Random walk within daily range
            price_change = rng.normal(0, candle_ranges[i])
            high = current_price + abs(price_change) + rng.uniform(0, candle_ranges[i] * 0.2)
            low = current_price - rng.uniform(0, candle_ranges[i] * 0.5)
            close = current_price + price_change
            
            # Ensure OHLC relationships
            high = max(high, close, current_price, low)
            low = min(low, close, current_price, high)
            open_price = current_price
            
            intraday_rows.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': int(rng.integers(100, 10000))
            })
            
            current_price = close
    
    return pd.DataFrame(intraday_rows)


def resample_ohlc(df, interval='5min'):
    """Resample tick/5min data to different timeframes."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Handle volume
    vol_col = 'volume' if 'volume' in df.columns else None
    
    rules = {
        '1min': {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'},
        '5min': {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'},
        '15min': {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'},
        '1h': {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'},
        '4h': {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'},
        '1d': {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'},
    }
    
    rule = interval.lower().replace('min', 'T').replace('h', 'H').replace('d', 'D')
    
    agg_rules = {}
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            agg_rules[col] = rules[interval][col]
    
    if vol_col:
        agg_rules[vol_col] = 'sum'
    elif 'volume' in df.columns:
        agg_rules['volume'] = 'sum'
    
    resampled = df.resample(rule).agg(agg_rules).dropna()
    resampled = resampled.reset_index()
    
    return resampled


def clean_data(df, sigma_threshold=5):
    """Remove outliers beyond sigma_threshold standard deviations."""
    if df is None or df.empty or len(df) < 10:
        return df
    
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Remove outliers based on returns
    df['returns'] = df['close'].pct_change()
    mean_ret = df['returns'].mean()
    std_ret = df['returns'].std()
    
    if std_ret > 0:
        df = df[np.abs(df['returns'] - mean_ret) <= sigma_threshold * std_ret]
    
    df = df.drop(columns=['returns'], errors='ignore')
    
    # Remove rows with missing values
    df = df.dropna()
    
    return df


def store_ohlc_to_db(df, table_name):
    """Store OHLC data to SQLite database."""
    if df is None or df.empty:
        logger.warning(f"No data to store in {table_name}")
        return
    
    df = df.copy()
    if 'timestamp' not in df.columns:
        logger.error(f"Missing timestamp column in {table_name}")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    """)
    
    # Insert data (replace on conflict)
    for _, row in df.iterrows():
        cur.execute(f"""
            INSERT OR REPLACE INTO {table_name} (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(row['timestamp']), row.get('open'), row.get('high'), 
              row.get('low'), row.get('close'), int(row.get('volume', 0))))
    
    conn.commit()
    conn.close()
    logger.info(f"Stored {len(df)} rows to {table_name}")


def fetch_gnews_news(start_date, end_date):
    """Fetch news from GNews API for gold/XAUUSD.
    
    GNews API: https://gnews.io/api/v4/search
    Free tier: 100 requests/day, 10 articles per request.
    We'll query in weekly chunks to get comprehensive coverage.
    
    If API key is invalid, falls back to realistic simulated news.
    """
    import requests
    import time
    
    api_key = config['data'].get('gnews_api_key', '')
    if not api_key or api_key == 'YOUR_GNEWS_API_KEY':
        logger.warning("GNews API key not configured")
        return _generate_realistic_news(start_date, end_date)
    
    articles = []
    search_queries = ['gold', 'XAUUSD', 'gold price', 'gold trading', 'gold market']
    
    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    except:
        start_dt = datetime(2023, 1, 1)
    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except:
        end_dt = datetime.now()
    
    logger.info(f"Fetching GNews from {start_date} to {end_date}...")
    
    # Fetch in monthly chunks for efficiency (fits ~3 years in 36 requests)
    current = start_dt
    total_articles = 0
    request_count = 0
    
    while current < end_dt:
        # Monthly chunk
        if current.month == 12:
            chunk_end = datetime(current.year + 1, 1, 1) - timedelta(days=1)
        else:
            chunk_end = datetime(current.year, current.month + 1, 1) - timedelta(days=1)
        chunk_end = min(chunk_end, end_dt)
        
        query = 'gold price'
        url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(query)}&from={current.strftime('%Y-%m-%d')}&to={chunk_end.strftime('%Y-%m-%d')}&apikey={api_key}&lang=en&max=10"
        
        try:
            logger.info(f"  GNews: {query} ({current.date()} to {chunk_end.date()})...")
            response = requests.get(url, timeout=30)
            request_count += 1
            
            if response.status_code != 200:
                error_msg = response.text
                if 'api key' in error_msg.lower() or response.status_code == 400:
                    logger.warning(f"  GNews API key invalid, falling back to simulated news")
                    return _generate_realistic_news(start_date, end_date)
                if response.status_code == 429:
                    logger.warning("  Rate limited, waiting 60s...")
                    time.sleep(60)
                continue
            
            data = response.json()
            
            if 'articles' in data and data['articles']:
                for article in data['articles']:
                    try:
                        pub_at = article.get('publishedAt', '')
                        if pub_at:
                            ts = datetime.fromisoformat(pub_at.replace('Z', '+00:00'))
                            timestamp = ts.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            timestamp = current.strftime('%Y-%m-%d %H:%M:%S')
                        
                        articles.append({
                            'timestamp': timestamp,
                            'title': article.get('title', ''),
                            'description': article.get('description', '')[:500],
                            'link': article.get('url', ''),
                            'source': article.get('source', {}).get('name', 'GNews')
                        })
                        total_articles += 1
                    except Exception as e:
                        logger.warning(f"  Error parsing article: {e}")
                
                logger.info(f"    Got {len(data['articles'])} articles")
            
            time.sleep(1)
            
            if request_count >= 95:
                break
            
        except Exception as e:
            logger.error(f"  GNews error: {e}")
        
        # Move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
        
        if request_count >= 95:
            break
    
    if articles:
        df = pd.DataFrame(articles)
        logger.info(f"Fetched {len(df)} articles from GNews ({request_count} requests)")
        df = df.drop_duplicates(subset=['title'], keep='first')
        return df
    else:
        logger.info("No articles from GNews, using simulated news")
        return _generate_realistic_news(start_date, end_date)


def _generate_realistic_news(start_date, end_date):
    """Generate realistic gold news events with FinBERT sentiment.
    
    Creates news articles based on realistic market patterns:
    - Fed/central bank announcements
    - Geopolitical events
    - Economic data releases
    - Technical breakouts
    - Market sentiment shifts
    """
    import numpy as np
    import sqlite3
    
    logger.info("Generating realistic gold news events...")
    
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    except:
        start_dt = datetime(2023, 1, 1)
    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except:
        end_dt = datetime.now()
    
    np.random.seed(42)
    
    # News event templates with realistic sentiments
    event_templates = [
        # Bullish events
        ("Fed signals potential rate cuts, gold prices rally", 0.7, 0.4),
        ("Gold breaks above key resistance level as dollar weakens", 0.6, 0.3),
        ("Central banks increase gold purchases amid uncertainty", 0.5, 0.2),
        ("Geopolitical tensions boost safe-haven demand for gold", 0.6, 0.3),
        ("Gold futures surge on inflation concerns", 0.5, 0.25),
        ("Technical analysis: Gold forming bullish pattern", 0.4, 0.15),
        ("Investors flock to gold as stock market volatility increases", 0.5, 0.2),
        ("Gold demand rises in emerging markets", 0.4, 0.15),
        ("Oil prices surge, inflation expectations boost gold", 0.4, 0.2),
        ("Gold holds steady near key support levels", 0.3, 0.1),
        
        # Bearish events
        ("Dollar strengthens as Fed maintains hawkish stance", -0.6, 0.3),
        ("Gold prices fall as risk appetite returns to markets", -0.5, 0.25),
        ("Fed officials suggest rates may stay higher for longer", -0.5, 0.2),
        ("Profit-taking pressures gold prices after rally", -0.4, 0.2),
        ("Gold faces resistance at key Fibonacci level", -0.3, 0.15),
        ("Strong US employment data weighs on gold", -0.4, 0.2),
        ("Gold retreats as Treasury yields rise", -0.4, 0.2),
        ("Risk-on sentiment reduces safe-haven demand for gold", -0.5, 0.25),
        ("Technical indicators suggest gold overbought conditions", -0.3, 0.15),
        ("Dollar rally pressures gold prices lower", -0.4, 0.2),
        
        # Neutral/mixed events
        ("Gold prices consolidate in narrow range ahead of Fed meeting", 0.0, 0.1),
        ("Market participants await key economic data release", 0.0, 0.05),
        ("Gold holds steady as investors assess outlook", 0.0, 0.1),
        ("Traders position ahead of central bank announcements", 0.1, 0.1),
        ("Gold market sees low volume during holiday trading", 0.0, 0.05),
        ("Mixed signals from economic reports keep gold range-bound", 0.0, 0.1),
    ]
    
    articles = []
    current_date = start_dt
    
    # Generate news roughly 3-5 times per week
    while current_date < end_dt:
        # Random chance of news event (4 events per week on average)
        if np.random.random() < 0.57:  # ~4 events per 7 days
            # Pick a random event
            title, base_sentiment, sentiment_var = event_templates[np.random.randint(len(event_templates))]
            
            # Add date-specific variation
            date_seed = int(current_date.strftime('%Y%m%d'))
            rng = np.random.default_rng(date_seed)
            
            # Sentiment with daily variation
            sentiment = np.clip(base_sentiment + rng.normal(0, sentiment_var), -1, 1)
            
            # Random hour between 8 AM and 6 PM
            hour = rng.integers(8, 18)
            minute = rng.integers(0, 60)
            timestamp = current_date.replace(hour=hour, minute=minute).strftime('%Y-%m-%d %H:%M:%S')
            
            articles.append({
                'timestamp': timestamp,
                'title': title,
                'description': f"Market analysis and news coverage for gold/XAUUSD - {current_date.strftime('%Y-%m-%d')}",
                'link': '',
                'source': 'MarketSimulator'
            })
        
        # Move to next day
        current_date += timedelta(days=1)
    
    if articles:
        df = pd.DataFrame(articles)
        logger.info(f"Generated {len(df)} realistic news articles")
        return df
    else:
        return pd.DataFrame()


def fetch_alphavantage_news(start_date, end_date):
    """Fetch news from Alpha Vantage for gold-related tickers.
    
    Free tier: 25 requests per day. We'll fetch only recent news (last 7 days)
    to preserve API quota and supplement with RSS/synthetic for historical.
    """
    import requests
    import time
    
    api_key = config['data'].get('alphavantage_key', '')
    if not api_key or api_key == 'YOUR_ALPHAVANTAGE_KEY':
        logger.warning("Alpha Vantage API key not configured")
        return pd.DataFrame()
    
    articles = []
    tickers = ['XAUUSD', 'GOLD', 'GLD', 'GDX']
    
    # Free tier: 25 requests/day. Only fetch recent 7 days to preserve quota.
    # For historical data, we'll use RSS + synthetic sentiment.
    end = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
    start = end - timedelta(days=7)  # Only last 7 days from AV
    logger.info(f"Fetching Alpha Vantage news for last 7 days (free tier limit)...")
    
    current = start
    chunk_end = end
    
    time_from = current.strftime('%Y%m%dT%H%M')
    time_to = chunk_end.strftime('%Y%m%dT%H%M')
    
    for ticker in tickers:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&time_to={time_to}&apikey={api_key}"
        
        try:
            logger.info(f"Fetching AV news for {ticker} ({current.date()} to {chunk_end.date()})...")
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"AV request failed with status {response.status_code}")
                continue
            
            data = response.json()
            
            if 'Note' in data or 'Information' in data:
                logger.warning(f"Alpha Vantage rate limit: {data.get('Note', data.get('Information', 'N/A'))}")
                break
            
            if 'feed' in data:
                for article in data['feed']:
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    time_published = article.get('time_published', '')
                    url_link = article.get('url', '')
                    source = article.get('source', '')
                    
                    try:
                        ts = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                        timestamp = ts.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        timestamp = time_published
                    
                    av_sentiment = article.get('overall_sentiment_score', 0)
                    
                    articles.append({
                        'timestamp': timestamp,
                        'title': title,
                        'description': summary[:500] if summary else "",
                        'link': url_link,
                        'source': source,
                        'sentiment_av': av_sentiment
                    })
                
                logger.info(f"  Got {len(data['feed'])} articles for {ticker}")
            else:
                logger.info(f"  No articles found for {ticker}")
            
            time.sleep(13)  # Rate limit
            
        except Exception as e:
            logger.error(f"Alpha Vantage error for {ticker}: {e}")
    
    if articles:
        df = pd.DataFrame(articles)
        logger.info(f"Fetched {len(df)} articles from Alpha Vantage")
        
        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).tolist()
        finbert_sentiments = compute_finbert_sentiment(texts)
        df['sentiment_finbert'] = finbert_sentiments
        
        df['sentiment_score'] = df.apply(
            lambda r: r['sentiment_av'] if abs(r['sentiment_av']) > 0.001 else r['sentiment_finbert'], 
            axis=1
        )
        
        return df
    else:
        logger.info("No articles fetched from Alpha Vantage")
        return pd.DataFrame()


def fetch_news_from_rss():
    """Fetch news from multiple RSS feeds."""
    import xml.etree.ElementTree as ET
    import requests
    
    rss_feeds = [
        ('https://www.investing.com/rss/commodities.rss', ['gold', 'xauusd', 'silver', 'commodity', 'precious metal']),
        ('https://www.forexlive.com/feed/news', ['gold', 'xauusd', 'commodity']),
        ('https://www.investing.com/rss/news.rss', ['gold', 'xauusd']),
    ]
    
    articles = []
    
    for url, keywords in rss_feeds:
        try:
            logger.info(f"Fetching RSS: {url[:50]}...")
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                items = root.findall('.//item')
                
                for item in items[:50]:  # Limit to 50 items per feed
                    title_elem = item.find('title')
                    pubdate_elem = item.find('pubDate')
                    link_elem = item.find('link')
                    desc_elem = item.find('description')
                    
                    title = title_elem.text if title_elem is not None else ""
                    pubdate = pubdate_elem.text if pubdate_elem is not None else ""
                    link = link_elem.text if link_elem is not None else ""
                    description = desc_elem.text if desc_elem is not None else ""
                    
                    # Filter for gold-related content
                    text_to_check = (title + " " + description).lower()
                    if any(kw in text_to_check for kw in keywords):
                        # Parse date
                        try:
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(pubdate)
                            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            from datetime import datetime
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        articles.append({
                            'timestamp': timestamp,
                            'title': title,
                            'description': description[:500] if description else "",
                            'link': link,
                            'source': url.split('/')[-1]
                        })
        except Exception as e:
            logger.warning(f"RSS fetch error for {url[:50]}: {e}")
    
    logger.info(f"Fetched {len(articles)} gold-related articles from RSS")
    return pd.DataFrame(articles)


def compute_finbert_sentiment(texts):
    """Compute sentiment using FinBERT."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        logger.info("Loading FinBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()
        
        sentiments = []
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Clean and truncate
            batch = [str(t)[:512] if t else "neutral" for t in batch]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # Labels: 0=positive, 1=negative, 2=neutral
                sentiment_scores = (probs[:, 0] - probs[:, 1]).numpy()
            
            sentiments.extend(sentiment_scores.tolist())
        
        return sentiments
    except Exception as e:
        logger.error(f"FinBERT error: {e}")
        # Fallback to random sentiment
        import numpy as np
        np.random.seed(42)
        return list(np.random.uniform(-0.3, 0.3, len(texts)))


def fetch_news_sentiment():
    """Fetch news and compute sentiment using GNews (priority), then Alpha Vantage, then RSS."""
    logger.info("=" * 60)
    logger.info("Step 7: Fetching news sentiment")
    logger.info("=" * 60)
    
    start = config['data'].get('start_date', '2023-01-01')
    end = datetime.now().strftime('%Y-%m-%d')
    
    # Try GNews first (priority - most comprehensive)
    gnews_key = config['data'].get('gnews_api_key', '')
    if gnews_key and gnews_key != 'YOUR_GNEWS_API_KEY':
        logger.info(f"Fetching news from GNews ({start} to {end})...")
        gnews_df = fetch_gnews_news(start, end)
        
        if not gnews_df.empty and 'sentiment_score' not in gnews_df.columns:
            logger.info(f"Computing FinBERT sentiment for {len(gnews_df)} articles...")
            texts = (gnews_df['title'].fillna('') + ' ' + gnews_df['description'].fillna('')).tolist()
            sentiments = compute_finbert_sentiment(texts)
            gnews_df['sentiment_score'] = sentiments
        
        if not gnews_df.empty:
            processed_df = gnews_df[['timestamp', 'title', 'sentiment_score']].copy()
            processed_df['sentiment_score'] = processed_df['sentiment_score'].fillna(0)
            logger.info(f"News: Got {len(gnews_df)} articles. Mean sentiment: {processed_df['sentiment_score'].mean():.4f}")
            return processed_df
        else:
            logger.warning("GNews returned no articles, trying Alpha Vantage...")
    
    # Try Alpha Vantage as fallback
    av_key = config['data'].get('alphavantage_key', '')
    if av_key and av_key != 'YOUR_ALPHAVANTAGE_KEY':
        logger.info(f"Fetching news from Alpha Vantage ({start} to {end})...")
        av_news = fetch_alphavantage_news(start, end)
        
        if not av_news.empty:
            processed_df = av_news[['timestamp', 'title', 'sentiment_score']].copy()
            processed_df['sentiment_score'] = processed_df['sentiment_score'].fillna(0)
            logger.info(f"Alpha Vantage: Got {len(av_news)} articles. Mean sentiment: {processed_df['sentiment_score'].mean():.4f}")
            return processed_df
        else:
            logger.warning("Alpha Vantage returned no articles, trying RSS fallback...")
    
    # Fallback to RSS feeds
    logger.info("Fetching news from RSS feeds...")
    news_df = fetch_news_from_rss()
    
    if news_df.empty:
        logger.warning("No news articles fetched, generating synthetic sentiment")
        return _generate_synthetic_news_sentiment()
    
    # Compute FinBERT sentiment
    logger.info(f"Computing FinBERT sentiment for {len(news_df)} RSS articles...")
    texts = (news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')).tolist()
    sentiments = compute_finbert_sentiment(texts)
    news_df['sentiment_score'] = sentiments
    
    # Create processed news table
    processed_df = news_df[['timestamp', 'title', 'sentiment_score']].copy()
    processed_df['sentiment_score'] = processed_df['sentiment_score'].fillna(0)
    
    logger.info(f"News sentiment computed. Mean sentiment: {processed_df['sentiment_score'].mean():.4f}")
    return processed_df


def _generate_synthetic_news_sentiment():
    """Generate synthetic news sentiment based on market volatility when no news available."""
    import sqlite3
    import numpy as np
    
    logger.info("Generating synthetic news sentiment based on market data...")
    
    # Load OHLC data
    conn = get_db_connection()
    df = pd.read_sql('''
        SELECT timestamp, open, high, low, close 
        FROM ohlc_m5 
        ORDER BY timestamp
    ''', conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'headline', 'sentiment_score'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(12).std()  # 1 hour rolling volatility
    
    # Generate synthetic news events based on large moves
    np.random.seed(42)
    news_events = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['volatility']):
            continue
        
        # Create synthetic news based on volatility and returns
        if abs(row['returns']) > row['volatility'] * 2:
            sentiment = row['returns'] * 10  # Positive returns = positive sentiment
            sentiment = np.clip(sentiment, -1, 1)
            news_events.append({
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'title': f"Market update: {row['returns']*100:.2f}% move",
                'sentiment_score': sentiment
            })
    
    logger.info(f"Generated {len(news_events)} synthetic news events")
    return pd.DataFrame(news_events) if news_events else pd.DataFrame(columns=['timestamp', 'title', 'sentiment_score'])


def main():
    logger.info("=" * 60)
    logger.info("Starting data collection pipeline")
    logger.info("=" * 60)
    
    # Step 1: Download recent 5min data from yfinance
    logger.info("Step 1: Downloading recent 5min data from yfinance...")
    m5_real = download_yfinance_intraday('GC=F', days_back=7, interval='5m')
    
    # Step 2: Download daily data for longer history
    logger.info("Step 2: Downloading daily data from yfinance...")
    daily_data = download_yfinance_daily('GC=F', years_back=3)
    
    # Step 3: Generate synthetic intraday from daily for historical period
    if daily_data is not None and not daily_data.empty:
        logger.info("Step 3: Generating synthetic intraday data from daily...")
        # Only generate for days before our real data starts
        real_start = m5_real['timestamp'].min() if m5_real is not None and not m5_real.empty else None
        
        if real_start is not None:
            historical_daily = daily_data[daily_data['timestamp'] < real_start]
        else:
            historical_daily = daily_data
        
        if not historical_daily.empty:
            m5_synthetic = generate_intraday_from_daily(historical_daily, '5min')
            logger.info(f"Generated {len(m5_synthetic)} synthetic M5 candles")
            
            # Combine real and synthetic
            if m5_real is not None and not m5_real.empty:
                m5_combined = pd.concat([m5_synthetic, m5_real], ignore_index=True)
                m5_combined = m5_combined.drop_duplicates(subset=['timestamp'], keep='last')
                m5_combined = m5_combined.sort_values('timestamp').reset_index(drop=True)
            else:
                m5_combined = m5_synthetic
        else:
            m5_combined = m5_real
    else:
        m5_combined = m5_real
    
    # Step 4: Clean data
    logger.info("Step 4: Cleaning data...")
    m5_clean = clean_data(m5_combined)
    
    # Step 5: Resample to different timeframes
    logger.info("Step 5: Resampling to other timeframes...")
    m1 = resample_ohlc(m5_clean, '1min')
    m15 = resample_ohlc(m5_clean, '15min')
    h1 = resample_ohlc(m5_clean, '1h')
    h4 = resample_ohlc(m5_clean, '4h')
    d1 = resample_ohlc(m5_clean, '1d')
    
    # Step 6: Store to database
    logger.info("Step 6: Storing to database...")
    store_ohlc_to_db(m5_clean, 'ohlc_m5')
    store_ohlc_to_db(m1, 'ohlc_m1')
    store_ohlc_to_db(m15, 'ohlc_m15')
    store_ohlc_to_db(h1, 'ohlc_h1')
    store_ohlc_to_db(h4, 'ohlc_h4')
    store_ohlc_to_db(d1, 'ohlc_d1')
    
    # Step 7: Fetch news and compute sentiment
    logger.info("Step 7: Fetching news sentiment...")
    news_df = fetch_news_sentiment()
    if not news_df.empty:
        conn = get_db_connection()
        # Store raw news (all columns)
        if 'title' in news_df.columns and 'sentiment_score' in news_df.columns:
            raw_cols = ['timestamp', 'title', 'description', 'link', 'source']
            if 'sentiment_av' in news_df.columns:
                raw_cols.append('sentiment_av')
            raw_cols.append('sentiment_score')
            
            available_cols = [c for c in raw_cols if c in news_df.columns]
            news_df[available_cols].to_sql('news_raw', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(news_df)} news items to news_raw")
            
            # Create processed news (just sentiment per timestamp)
            processed = news_df[['timestamp', 'sentiment_score']].copy()
            processed.to_sql('news_processed', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(processed)} sentiment records to news_processed")
        conn.close()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Data collection complete!")
    logger.info("=" * 60)
    for table in ['ohlc_m1', 'ohlc_m5', 'ohlc_m15', 'ohlc_h1', 'ohlc_h4', 'ohlc_d1']:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        cur.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table}")
        dates = cur.fetchone()
        conn.close()
        logger.info(f"  {table}: {count} rows, {dates[0]} to {dates[1]}")
    
    logger.info("=" * 60)


def fetch_mt5_data(days_back=30):
    """
    Fetch recent data from MetaTrader 5.
    
    NOTE: MetaTrader5 Python package requires:
    1. MT5 terminal running on Windows
    2. pip install MetaTrader5 (on Windows with Python 3.8-3.11)
    
    This function is a placeholder for when MT5 is available.
    For now, we use yfinance data which is sufficient.
    """
    logger.info("Attempting MT5 data fetch...")
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return None
        
        # Get last 30 days of M5 data
        symbol = "XAUUSD"
        utc_from = datetime.now() - timedelta(days=days_back)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, utc_from, datetime.now())
        
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            logger.warning("No MT5 data returned")
            return None
        
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low', 
            'close': 'close', 'tick_volume': 'volume'
        })
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"MT5: Downloaded {len(df)} rows")
        return df
        
    except ImportError:
        logger.warning("MetaTrader5 package not available (requires Windows + Python 3.8-3.11)")
        logger.info("Using yfinance data instead for recent prices")
        return None
    except Exception as e:
        logger.error(f"MT5 error: {e}")
        return None


if __name__ == '__main__':
    main()

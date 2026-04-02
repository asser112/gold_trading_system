#!/usr/bin/env python3
"""
Ensemble and Decision Logic
- Combines XGBoost, Transformer, RL using logistic regression meta-learner
- Applies filters (trend, volatility, news)
- Outputs trading signal with confidence
"""
import pandas as pd
import numpy as np
import joblib
import torch
import yaml
import logging
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from stable_baselines3 import PPO
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec = importlib.util.spec_from_file_location('train_transformer', 'scripts/04_train_transformer.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TransformerModel = mod.TransformerModel

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)

def load_models():
    """Load all three models."""
    xgb_model = joblib.load('models/xgboost/xgboost_best.pkl')
    # Transformer
    # Need input_dim; we'll read from saved scaler
    scaler = joblib.load('models/scalers/robust_scaler.pkl')
    input_dim = len(scaler.scale_)
    transformer_model = TransformerModel(
        input_dim=input_dim,
        d_model=config['models']['transformer']['d_model'],
        nhead=config['models']['transformer']['nhead'],
        num_layers=config['models']['transformer']['num_layers'],
        dim_feedforward=config['models']['transformer']['dim_feedforward'],
        dropout=config['models']['transformer']['dropout']
    )
    transformer_model.load_state_dict(torch.load('models/transformer/best_model.pth', map_location='cpu'))
    transformer_model.eval()
    # RL - use latest checkpoint or final_model
    import glob
    rl_paths = glob.glob('models/rl_agent/rl_model_*_steps.zip')
    rl_model = None
    if os.path.exists('models/rl_agent/final_model.zip'):
        rl_path = 'models/rl_agent/final_model.zip'
        rl_model = PPO.load(rl_path)
    elif rl_paths:
        rl_path = sorted(rl_paths, key=lambda x: int(x.split('rl_model_')[1].split('_')[0]))[-1]
        logger.info(f"Using RL checkpoint: {rl_path}")
        rl_model = PPO.load(rl_path)
    else:
        logger.warning("No RL model found, skipping RL in ensemble")
    return xgb_model, transformer_model, rl_model

def generate_predictions(df, xgb_model, transformer_model, rl_model, all_features, xgb_features, seq_len=60):
    """Generate predictions for each sample in df (row-wise)."""
    X_all = df[all_features].values
    X_xgb = df[xgb_features].values
    X_rl = df[xgb_features].values  # RL was trained without ATR
    n = len(X_all)
    # XGBoost predictions (probabilities for classes -1,0,1)
    xgb_probs = xgb_model.predict_proba(X_xgb)  # shape (n, 3)
    # Transformer predictions (require sequences)
    transformer_probs = np.zeros((n, 3))
    for i in range(seq_len, n):
        seq = X_all[i-seq_len:i].reshape(1, seq_len, -1)
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        with torch.no_grad():
            out = transformer_model(seq_tensor).numpy()
        transformer_probs[i] = out
    # RL predictions: need to construct observation for each step
    window = config['models']['rl']['window']
    rl_actions = np.zeros(n)
    for i in range(window, n):
        obs_features = X_rl[i-window:i].flatten()
        obs = np.concatenate([obs_features, [0]])
        action, _ = rl_model.predict(obs, deterministic=True)
        rl_actions[i] = action
    meta_features = np.hstack([xgb_probs, transformer_probs, rl_actions.reshape(-1,1)])
    return meta_features

def train_meta_learner():
    """Train a logistic regression meta-learner on out-of-sample predictions."""
    df = pd.read_parquet('data/processed/features_target_m5.parquet')
    df = df.dropna()
    # All features (matching current feature set)
    all_features = ['rsi', 'atr', 'ema20', 'ema50', 'vwap', 'bb_upper', 'bb_middle', 'bb_lower', 
                    'bb_width', 'adx', 'order_block', 'fvg_distance', 'liquidity_zone', 
                    'sweep', 'sentiment_score', 'hour', 'day_of_week', 
                    'session_Asian', 'session_London', 'session_NY']
    # XGB uses all features (same as transformer/RL now)
    xgb_features = all_features.copy()
    all_features_filtered = [c for c in all_features if c in df.columns]
    xgb_features_filtered = [c for c in xgb_features if c in df.columns]
    X = df[all_features_filtered]
    y = df['target'] + 1  # convert to 0,1,2

    # Split chronologically
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Load models
    xgb_model, transformer_model, rl_model = load_models()

    # Generate predictions for training set
    logger.info("Generating predictions for training set...")
    train_meta = generate_predictions(X_train, xgb_model, transformer_model, rl_model, 
                                       all_features_filtered, xgb_features_filtered)
    # Remove rows where any model couldn't predict (first seq_len rows)
    seq_len = config['models']['transformer']['seq_len']
    window = config['models']['rl']['window']
    min_len = max(seq_len, window)
    valid_idx = slice(min_len, len(train_meta))
    train_meta = train_meta[valid_idx]
    y_train_meta = y_train.iloc[valid_idx].values

    logger.info("Generating predictions for test set...")
    test_meta = generate_predictions(X_test, xgb_model, transformer_model, rl_model,
                                      all_features_filtered, xgb_features_filtered)
    test_meta = test_meta[min_len:]
    y_test_meta = y_test.iloc[min_len:].values

    # Train meta-learner
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(train_meta, y_train_meta)
    preds = meta_model.predict(test_meta)
    acc = accuracy_score(y_test_meta, preds)
    logger.info(f"Meta-learner test accuracy: {acc:.4f}")

    # Save meta-learner
    joblib.dump(meta_model, 'models/ensemble/meta_learner.pkl')
    logger.info("Ensemble meta-learner saved.")

# For live signal generation, we need to maintain rolling buffers
# We'll create a class that caches recent data
class SignalGenerator:
    def __init__(self):
        self.xgb_model = joblib.load('models/xgboost/xgboost_best.pkl')
        self.transformer_model = self._load_transformer()
        self.rl_model = self._load_rl_model()
        self.meta_model = joblib.load('models/ensemble/meta_learner.pkl')
        self.feature_cols = None  # will be set later
        self.seq_len = config['models']['transformer']['seq_len']
        self.window = config['models']['rl']['window']
        self.buffer = []  # list of feature vectors
        self.position = 0  # current position (0,1,-1)
    
    def _load_rl_model(self):
        if os.path.exists('models/rl_agent/final_model.zip'):
            return PPO.load('models/rl_agent/final_model.zip')
        rl_paths = glob.glob('models/rl_agent/rl_model_*_steps.zip')
        if rl_paths:
            latest = sorted(rl_paths, key=lambda x: int(x.split('rl_model_')[1].split('_')[0]))[-1]
            logger.info(f"Loading RL from: {latest}")
            return PPO.load(latest)
        return None

    def _load_transformer(self):
        scaler = joblib.load('models/scalers/robust_scaler.pkl')
        input_dim = len(scaler.scale_)
        model = TransformerModel(
            input_dim=input_dim,
            d_model=config['models']['transformer']['d_model'],
            nhead=config['models']['transformer']['nhead'],
            num_layers=config['models']['transformer']['num_layers'],
            dim_feedforward=config['models']['transformer']['dim_feedforward'],
            dropout=config['models']['transformer']['dropout']
        )
        model.load_state_dict(torch.load('models/transformer/best_model.pth', map_location='cpu'))
        model.eval()
        return model

    def update(self, current_features, news_aggregate):
        """Update buffer with new feature vector (including news)."""
        # current_features is a Series/array with all features
        if self.feature_cols is None:
            self.feature_cols = list(current_features.index)
        # Add news sentiment to features if not already present
        # Assuming news_aggregate is a scalar, we'll append it to features
        # But features should already include news_sentiment column. We'll trust that.
        row = current_features.values
        self.buffer.append(row)
        # Keep only last (max(seq_len, window)) items
        max_len = max(self.seq_len, self.window)
        if len(self.buffer) > max_len:
            self.buffer.pop(0)

    def get_signal(self):
        """Generate signal based on current buffer."""
        if len(self.buffer) < max(self.seq_len, self.window):
            return 0, 0.0  # insufficient data

        # Prepare current data row (latest)
        current_row = np.array(self.buffer[-1]).reshape(1, -1)
        # XGBoost
        xgb_prob = self.xgb_model.predict_proba(current_row)[0]
        
        # Use XGBoost directly since meta-learner expects 3 features
        predicted_class = np.argmax(xgb_prob)
        confidence = xgb_prob[predicted_class]

        # Apply filters
        trend_ok = True
        if config['models']['ensemble']['trend_filter']:
            ema20 = current_row[0, self.feature_cols.index('ema20')] if 'ema20' in self.feature_cols else 0
            ema50 = current_row[0, self.feature_cols.index('ema50')] if 'ema50' in self.feature_cols else 0
            adx = current_row[0, self.feature_cols.index('adx')] if 'adx' in self.feature_cols else 0
            trend_ok = (ema20 > ema50) and (adx > 25)

        volatility_ok = True
        if config['models']['ensemble']['volatility_filter']:
            bb_width = current_row[0, self.feature_cols.index('bb_width')] if 'bb_width' in self.feature_cols else 0
            volatility_ok = bb_width > 0.001  # Minimum volatility threshold

        news_ok = True
        if config['models']['ensemble']['news_filter']:
            news = current_row[0, self.feature_cols.index('sentiment_score')] if 'sentiment_score' in self.feature_cols else 0
            news_ok = abs(news) < 0.5

        # Use lower threshold for XGBoost-only mode (0.60 instead of 0.97)
        xgb_threshold = 0.60
        if predicted_class != 1 and confidence > xgb_threshold and trend_ok and news_ok and volatility_ok:
            signal = 1 if predicted_class == 2 else -1 if predicted_class == 0 else 0
            return signal, confidence
        else:
            return 0, 0.0

if __name__ == '__main__':
    train_meta_learner()
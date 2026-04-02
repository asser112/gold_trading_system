#!/usr/bin/env python3
"""
Reinforcement Learning Agent Training
- Environment: last 20 candles + last 5-min news aggregates + current position
- Actions: hold, buy, sell
- Reward: Sharpe ratio (daily) or PnL normalized by ATR
- PPO with LSTM policy
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, data, feature_cols, news_cols, atr_col='atr', window=20, initial_balance=10000, commission=0.001, spread=0.0030):
        super().__init__()
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.news_cols = news_cols
        self.atr_col = atr_col
        self.window = window
        self.initial_balance = initial_balance
        self.commission = commission
        self.spread = spread
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        n_features = len(feature_cols) + len(news_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window * n_features + 1,), dtype=np.float32)
        self.current_step = window
        self.balance = initial_balance
        self.position = 0  # 0=none, 1=long, -1=short
        self.entry_price = 0
        self.daily_pnl = []
        self.current_day = None
        
        # Normalize features
        feat_cols = feature_cols + news_cols
        self.data[feat_cols] = self.data[feat_cols].replace([np.inf, -np.inf], np.nan)
        self.data[feat_cols] = self.data[feat_cols].fillna(0)
        self.feat_mean = self.data[feat_cols].mean()
        self.feat_std = self.data[feat_cols].std().replace(0, 1)
        self.data[feat_cols] = (self.data[feat_cols] - self.feat_mean) / self.feat_std
        
        # Replace inf/nan in close
        self.data['close'] = self.data['close'].replace([np.inf, -np.inf], np.nan).fillna(self.data['close'].mean())
        self.data['atr'] = self.data['atr'].replace([np.inf, -np.inf], np.nan).fillna(1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.daily_pnl = []
        self.current_day = self.data.index[self.current_step].date()
        return self._get_obs(), {}

    def step(self, action):
        price = self.data.iloc[self.current_step]['close']
        atr = self.data.iloc[self.current_step][self.atr_col]
        reward = 0
        terminated = False
        truncated = False

        # Execute action
        if action == 1:  # buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.balance -= self.commission * price
                self.balance -= self.spread * 30  # Spread cost: 30 pips
            elif self.position == -1:
                # Close short
                pnl = (self.entry_price - price)
                self.balance += pnl
                self.balance -= self.commission * price
                self.daily_pnl.append(pnl)
                # Open long
                self.entry_price = price
                self.balance -= self.commission * price
                self.balance -= self.spread * 30  # Spread cost
                self.position = 1
        elif action == 2:  # sell
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.balance -= self.commission * price
                self.balance -= self.spread * 30  # Spread cost
            elif self.position == 1:
                pnl = (price - self.entry_price)
                self.balance += pnl
                self.balance -= self.commission * price
                self.daily_pnl.append(pnl)
                self.entry_price = price
                self.balance -= self.commission * price
                self.balance -= self.spread * 30  # Spread cost
                self.position = -1
        # else hold: do nothing

        # Check if day changed
        current_date = self.data.index[self.current_step].date()
        if current_date != self.current_day:
            # End of day: compute reward as Sharpe ratio of daily PnL (or normalized PnL)
            if len(self.daily_pnl) > 0:
                daily_return = np.sum(self.daily_pnl) / self.initial_balance
                reward = daily_return * 100  # scale
            else:
                reward = 0
            self.daily_pnl = []
            self.current_day = current_date
        else:
            # Intraday reward: normalized PnL of open position
            if self.position != 0 and atr > 0:
                unrealized = (price - self.entry_price) if self.position == 1 else (self.entry_price - price)
                reward = unrealized / atr
            else:
                reward = 0

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            terminated = True

        if terminated and self.position != 0:
            final_price = self.data.iloc[-1]['close']
            pnl = (final_price - self.entry_price) if self.position == 1 else (self.entry_price - final_price)
            self.balance += pnl
            self.balance -= self.commission * final_price
            self.daily_pnl.append(pnl)

        obs = self._get_obs()
        info = {'balance': self.balance, 'position': self.position}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        start = self.current_step - self.window
        end = self.current_step
        window_data = self.data.iloc[start:end]
        # Combine features and news
        feat = window_data[self.feature_cols + self.news_cols].values.flatten()
        pos = np.array([self.position], dtype=np.float32)
        return np.concatenate([feat, pos])

def main():
    # Load features with target
    df = pd.read_parquet('data/processed/features_target_m5.parquet')
    
    # Get close price from database
    import sqlite3
    conn = sqlite3.connect('data/gold_trading.db')
    df_ohlc = pd.read_sql('SELECT timestamp, close FROM ohlc_m5 ORDER BY timestamp', conn, parse_dates=['timestamp'])
    conn.close()
    
    # Merge close price
    df_ohlc['timestamp'] = pd.to_datetime(df_ohlc['timestamp'])
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.merge(df_ohlc[['timestamp', 'close']], on='timestamp', how='left')
    df = df.set_index('timestamp')
    
    # Ensure sentiment_score exists
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0
    
    feature_cols = [c for c in df.columns if c not in ['target', 'close', 'atr', 'sentiment_score']]
    news_cols = ['sentiment_score']
    df = df[feature_cols + ['close', 'atr'] + news_cols]
    df = df.dropna()

    # Create environment with spread cost for Exness Standard (30 pips)
    spread_cost = config['backtest'].get('spread', 0.0030)
    env = TradingEnv(df, feature_cols, news_cols, atr_col='atr', window=config['models']['rl']['window'], spread=spread_cost)
    env = DummyVecEnv([lambda: env])

    # PPO with custom network
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    model = PPO('MlpPolicy', env, verbose=1,
                learning_rate=config['models']['rl']['learning_rate'],
                n_steps=config['models']['rl']['n_steps'],
                batch_size=config['models']['rl']['batch_size'],
                n_epochs=config['models']['rl']['n_epochs'],
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=policy_kwargs, seed=42)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='models/rl_agent/')
    total_timesteps = config['models']['rl']['episodes'] * config['models']['rl']['steps_per_episode']
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save('models/rl_agent/final_model')
    logger.info("RL training completed.")

if __name__ == '__main__':
    main()
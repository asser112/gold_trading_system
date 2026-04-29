#!/usr/bin/env python3
"""
LightGBM Training Module (Separate Pipeline — Approach 1)
- Loads features from data/processed/features_lgbm_m5.parquet
- Filters training data to London + NY sessions only (high-probability bars)
- Walk-forward validation with Optuna hyperparameter tuning
- Saves best model to models/lightgbm/lgbm_best.pkl
"""
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import yaml
import logging
from sklearn.metrics import f1_score, accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

LGBM_CFG = config.get('lightgbm', {})
SESSION_CFG = LGBM_CFG.get('session_filter', {})
LONDON_START = SESSION_CFG.get('london_start_utc', 8)
LONDON_END   = SESSION_CFG.get('london_end_utc', 17)
NY_START     = SESSION_CFG.get('ny_start_utc', 13)
NY_END       = SESSION_CFG.get('ny_end_utc', 21)


def filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows falling inside London or NY session hours."""
    hour = df.index.hour
    mask = ((hour >= LONDON_START) & (hour < LONDON_END)) | ((hour >= NY_START) & (hour < NY_END))
    filtered = df[mask]
    logger.info(
        f"Session filter: {len(df):,} → {len(filtered):,} bars "
        f"(London {LONDON_START}–{LONDON_END} UTC + NY {NY_START}–{NY_END} UTC)"
    )
    return filtered


def create_walk_forward_splits(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)
    return [(df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:])]


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
    )
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='weighted')


def main():
    logger.info("=" * 60)
    logger.info("LIGHTGBM TRAINING — START")
    logger.info("=" * 60)

    features_path = 'data/processed/features_lgbm_m5.parquet'
    if not os.path.exists(features_path):
        logger.error(f"Features file not found: {features_path}")
        logger.error("Run scripts/02b_feature_engineering_lgbm.py first.")
        sys.exit(1)

    df = pd.read_parquet(features_path).dropna()
    logger.info(f"Loaded {len(df):,} rows from {features_path}")
    logger.info(f"Target distribution (before filter): {df['target'].value_counts().to_dict()}")

    # Apply session filter — train only on London + NY bars
    df = filter_session(df)
    logger.info(f"Target distribution (after filter):  {df['target'].value_counts().to_dict()}")

    if df.empty:
        logger.error("No data after session filtering.")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in ('target',)]
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Map target: -1→0 (sell), 0→1 (hold), 1→2 (buy)
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_mapped'] = df['target'].map(label_map)

    splits = create_walk_forward_splits(df)
    n_trials = LGBM_CFG.get('n_trials', 50)

    all_f1, all_acc, best_models = [], [], []

    for i, (train_df, val_df, test_df) in enumerate(splits):
        logger.info(f"\nFold {i+1}/{len(splits)}")
        logger.info(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

        if len(test_df) < 20:
            logger.warning("  Skipping: not enough test samples")
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df['target_mapped'].values
        X_val   = val_df[feature_cols].values
        y_val   = val_df['target_mapped'].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df['target_mapped'].values

        if len(X_val) > 0:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True
            )
            best_params = study.best_params
            logger.info(f"  Best val F1: {study.best_value:.4f}")
        else:
            best_params = {
                'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.05,
                'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            }

        # Retrain on train + val combined
        X_full = np.vstack([X_train, X_val]) if len(X_val) > 0 else X_train
        y_full = np.concatenate([y_train, y_val]) if len(X_val) > 0 else y_train

        final_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            verbosity=-1,
            random_state=42,
            n_jobs=-1,
            **best_params
        )
        final_model.fit(X_full, y_full)

        # Store feature names so predict can use them later
        final_model._feature_cols = feature_cols

        preds = final_model.predict(X_test)
        test_f1  = f1_score(y_test, preds, average='weighted')
        test_acc = accuracy_score(y_test, preds)

        all_f1.append(test_f1)
        all_acc.append(test_acc)
        best_models.append((final_model, test_f1))

        logger.info(f"  Test F1: {test_f1:.4f}  Accuracy: {test_acc:.4f}")

        os.makedirs('models/lightgbm', exist_ok=True)
        joblib.dump(final_model, f'models/lightgbm/lgbm_fold_{i+1}.pkl')

    if not best_models:
        logger.error("No models trained.")
        sys.exit(1)

    best_model = max(best_models, key=lambda x: x[1])[0]
    os.makedirs('models/lightgbm', exist_ok=True)
    joblib.dump(best_model, 'models/lightgbm/lgbm_best.pkl')

    # Feature importance
    importances = best_model.feature_importances_
    importance_df = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    importance_df.to_csv('models/lightgbm/feature_importance.csv')

    logger.info(f"\nBest model saved → models/lightgbm/lgbm_best.pkl")
    logger.info(f"Best Test F1    : {max(all_f1):.4f}")
    logger.info(f"Avg  Test F1    : {np.mean(all_f1):.4f}")
    logger.info(f"Avg  Accuracy   : {np.mean(all_acc):.4f}")
    logger.info("\nTop 10 Features by Importance:")
    for feat, imp in importance_df.head(10).items():
        logger.info(f"  {feat}: {imp:.0f}")

    logger.info("=" * 60)
    logger.info("LIGHTGBM TRAINING — DONE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

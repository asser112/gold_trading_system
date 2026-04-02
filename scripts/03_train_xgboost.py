#!/usr/bin/env python3
"""
XGBoost Training Module
- Uses walk-forward validation
- Optuna hyperparameter tuning
- Saves model and feature importance
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import f1_score, accuracy_score
import joblib
import yaml
import logging
from datetime import datetime, timedelta

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)


def create_walk_forward_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Generate chronological splits based on data availability."""
    n = len(df)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_end = train_size
    val_end = train_size + val_size
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return [(train_df, val_df, test_df)]


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 3),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1e1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1e1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'seed': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    
    model = xgb.XGBClassifier(
        **params,
        n_estimators=n_estimators,
        early_stopping_rounds=20,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='weighted')


def main():
    logger.info("Starting XGBoost training...")
    
    df = pd.read_parquet('data/processed/features_target_m5.parquet')
    df = df.dropna()
    
    if df.empty:
        logger.error("No data available")
        return
    
    feature_cols = [c for c in df.columns if c not in ['target', 'target_binary', 'close', 'atr']]
    
    if len(feature_cols) == 0:
        logger.error("No features found")
        return
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    splits = create_walk_forward_splits(df)
    logger.info(f"Created {len(splits)} walk-forward splits")
    
    n_trials = config['models']['xgboost'].get('n_trials', 20)
    
    all_f1 = []
    all_acc = []
    best_models = []
    
    for i, (train_df, val_df, test_df) in enumerate(splits):
        logger.info(f"\nFold {i+1}/{len(splits)}:")
        logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        if len(test_df) < 20:
            logger.warning(f"  Skipping fold {i+1}: not enough test samples")
            continue
        
        X_train = train_df[feature_cols]
        y_train = train_df['target'].astype(int)
        X_val = val_df[feature_cols] if not val_df.empty else None
        y_val = val_df['target'].astype(int) if not val_df.empty else None
        X_test = test_df[feature_cols]
        y_test = test_df['target'].astype(int)
        
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2}) if y_val is not None else None
        y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})
        
        if X_val is not None and len(X_val) > 0:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(
                lambda trial: objective(trial, X_train, y_train_mapped, X_val, y_val_mapped),
                n_trials=n_trials,
                show_progress_bar=False
            )
            best_params = study.best_params
            best_value = study.best_value
            logger.info(f"  Best validation F1: {best_value:.4f}")
        else:
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'min_child_weight': 1,
                'n_estimators': 100
            }
        
        final_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=best_params.get('n_estimators', 100),
            seed=42,
            n_jobs=-1,
            verbosity=0,
            **{k: v for k, v in best_params.items() if k != 'n_estimators'}
        )
        
        if X_val is not None and len(X_val) > 0:
            X_train_full = pd.concat([X_train, X_val])
            y_train_full = pd.concat([y_train_mapped, y_val_mapped])
        else:
            X_train_full = X_train
            y_train_full = y_train_mapped
        
        final_model.fit(X_train_full, y_train_full, verbose=False)
        
        preds = final_model.predict(X_test)
        test_f1 = f1_score(y_test_mapped, preds, average='weighted')
        test_acc = accuracy_score(y_test_mapped, preds)
        
        all_f1.append(test_f1)
        all_acc.append(test_acc)
        best_models.append((final_model, test_f1))
        
        logger.info(f"  Test F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}")
        
        os.makedirs('models/xgboost', exist_ok=True)
        joblib.dump(final_model, f'models/xgboost/xgboost_fold_{i+1}.pkl')
    
    if best_models:
        best_model = max(best_models, key=lambda x: x[1])[0]
        joblib.dump(best_model, 'models/xgboost/xgboost_best.pkl')
        
        importance = best_model.feature_importances_
        feature_importance = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
        feature_importance.to_csv('models/xgboost/feature_importance.csv')
        
        logger.info(f"\nBest model saved with F1: {max(all_f1):.4f}")
        logger.info(f"Average F1 across folds: {np.mean(all_f1):.4f}")
        
        logger.info("\nTop 10 Features by Importance:")
        for feat, imp in feature_importance.head(10).items():
            logger.info(f"  {feat}: {imp:.4f}")
    
    logger.info("XGBoost training completed.")


if __name__ == '__main__':
    main()

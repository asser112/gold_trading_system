#!/usr/bin/env python3
"""
Main orchestration script for the Gold trading system.
Runs all steps sequentially with error handling and logging.
"""
import os
import sys
import yaml
import logging
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

os.makedirs('logs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models/xgboost', exist_ok=True)
os.makedirs('models/transformer', exist_ok=True)
os.makedirs('models/rl_agent', exist_ok=True)
os.makedirs('models/ensemble', exist_ok=True)
os.makedirs('models/scalers', exist_ok=True)
os.makedirs('backtest_reports', exist_ok=True)

_log_file = config['logging']['file']
if not os.path.isabs(_log_file):
    _log_file = os.path.join(SCRIPT_DIR, _log_file)

logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_script(script_name, required=True):
    """Run a Python script and log output."""
    logger.info(f"{'='*50}")
    logger.info(f"Running {script_name}...")
    logger.info(f"{'='*50}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=7200
        )
        
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-20:]:
                logger.info(f"  {line}")
        
        logger.info(f"{script_name} completed successfully.")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{script_name} failed with return code {e.returncode}")
        if e.stdout:
            for line in e.stdout.strip().split('\n')[-10:]:
                logger.error(f"  OUT: {line}")
        if e.stderr:
            for line in e.stderr.strip().split('\n')[-10:]:
                logger.error(f"  ERR: {line}")
        if required:
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"{script_name} timed out after 2 hours")
        if required:
            return False
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {e}")
        if required:
            return False
    
    return True


def main():
    """Run the complete pipeline."""
    logger.info("="*60)
    logger.info("GOLD TRADING SYSTEM - Pipeline Starting")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Project root: {SCRIPT_DIR}")
    
    pipeline_steps = [
        ('Step 1: Data Collection', 'scripts/01_data_collection.py'),
        ('Step 2: Feature Engineering', 'scripts/02_feature_engineering.py'),
        ('Step 3: XGBoost Training', 'scripts/03_train_xgboost.py'),
        ('Step 4: Transformer Training', 'scripts/04_train_transformer.py'),
        ('Step 5: RL Agent Training', 'scripts/05_train_rl_agent.py'),
        ('Step 6: Ensemble Training', 'scripts/06_ensemble.py'),
        ('Step 7: Backtesting', 'scripts/08_backtester.py'),
    ]
    
    failed_steps = []
    
    for step_name, script_path in pipeline_steps:
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            failed_steps.append((step_name, "Script not found"))
            continue
            
        success = run_script(script_path, required=False)
        
        if not success:
            failed_steps.append((step_name, "Script failed"))
            logger.warning(f"{step_name} failed, continuing with next steps...")
    
    logger.info("="*60)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("="*60)
    
    if failed_steps:
        logger.warning(f"Failed steps: {len(failed_steps)}")
        for step, reason in failed_steps:
            logger.warning(f"  - {step}: {reason}")
    else:
        logger.info("All steps completed successfully!")
    
    logger.info(f"Finished at: {datetime.now().isoformat()}")
    
    return len(failed_steps) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

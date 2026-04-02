#!/usr/bin/env python3
"""
Monitoring Module
- Sends alerts via Telegram
- Generates dashboard data
- Optionally retrains monthly
"""
import yaml
import logging
import pandas as pd
from datetime import datetime
import subprocess
import sys
from telegram import Bot
from telegram.error import TelegramError

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)

bot = Bot(token=config['telegram']['api_token']) if config['telegram']['enabled'] else None

def send_alert(message):
    """Send Telegram message."""
    if bot:
        try:
            bot.send_message(chat_id=config['telegram']['chat_id'], text=message)
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")

def check_performance():
    """Check recent backtest results and send alert if performance drops."""
    try:
        with open('backtest_reports/report.txt', 'r') as f:
            report = f.read()
        # Parse some metrics (e.g., Sharpe ratio)
        # For demonstration, we just send the whole report
        send_alert(f"Latest backtest report:\n{report[-1000:]}")
    except Exception as e:
        logger.error(f"Error reading backtest report: {e}")

def monthly_retrain():
    """Run the full pipeline to retrain models."""
    logger.info("Starting monthly retraining...")
    send_alert("Monthly retraining started.")
    try:
        subprocess.run([sys.executable, 'run_pipeline.py'], check=True)
        send_alert("Monthly retraining completed successfully.")
    except subprocess.CalledProcessError as e:
        send_alert(f"Monthly retraining failed: {e}")

if __name__ == '__main__':
    # This script can be run on a schedule (e.g., cron)
    # For now, we just do a simple check
    check_performance()
    # If today is first of month, retrain
    if datetime.now().day == 1:
        monthly_retrain()
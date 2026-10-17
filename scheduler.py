"""
scheduler.py — Runs pipeline.py locally every day
-------------------------------------------------
Run this once in a terminal:
    export FRED_API_KEY=your_key   # free key: https://fred.stlouisfed.org/docs/api/api_key.html
    python scheduler.py

Runs the pipeline once on startup, then every day at RETRAIN_TIME.
Leave the terminal open to keep the scheduler running.
Press Ctrl+C to stop.
"""

import time
import schedule

import pipeline

RETRAIN_TIME = '08:00'  # Daily run time (24h format)


def run_pipeline():
    try:
        pipeline.run()
    except Exception:
        pass  # pipeline.run() already printed the error and wrote retrain_log.json


if __name__ == '__main__':
    print(f'{"=" * 55}')
    print('  Montréal Gas Price — Automated Pipeline Scheduler')
    print(f'{"=" * 55}')
    print(f'  Schedule : Daily at {RETRAIN_TIME}')
    print(f'  FRED key : {"set" if pipeline.FRED_API_KEY else "NOT SET — export FRED_API_KEY"}')
    print(f'{"=" * 55}\n')

    schedule.every().day.at(RETRAIN_TIME).do(run_pipeline)
    run_pipeline()

    while True:
        schedule.run_pending()
        time.sleep(60)

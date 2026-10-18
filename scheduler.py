"""
scheduler.py — Runs pipeline.py locally every day
-------------------------------------------------
Run this once in a terminal:
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
    except Exception as e:
        print(f'Pipeline failed: {e}')  # the published forecast is left untouched


if __name__ == '__main__':
    print(f'Montréal Gas Price — running pipeline daily at {RETRAIN_TIME}. Ctrl+C to stop.\n')
    schedule.every().day.at(RETRAIN_TIME).do(run_pipeline)
    run_pipeline()

    while True:
        schedule.run_pending()
        time.sleep(60)

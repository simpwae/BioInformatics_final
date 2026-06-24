"""
Lightweight run logger. Appends a one-line summary to results/run_log.jsonl
every time a training run completes. Keeps a machine-readable audit trail.
"""

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = ROOT / "results" / "run_log.jsonl"


def log_run(entry: dict):
    """Append a run summary to the run log."""
    entry["logged_at"] = datetime.utcnow().isoformat()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_run_log() -> list:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]

"""
Sentinel AI — Continuous Learning Flywheel
Phase 9: Data Flywheel

Stores human corrections in a SQLite database to later be exported as JSONL
for LoRA fine-tuning, closing the RLHF feedback loop.
"""
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FlywheelDB:
    """Stores user corrections and feedback to curate a fine-tuning dataset."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = "data/feedback.db"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(FlywheelDB, cls).__new__(cls)
                    cls._instance._init(db_path)
        return cls._instance

    def _init(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    username TEXT NOT NULL,
                    image_hash TEXT,
                    prompt TEXT NOT NULL,
                    original_answer TEXT NOT NULL,
                    user_correction TEXT,
                    rating INTEGER NOT NULL -- 1 for thumbs up, -1 for thumbs down
                )
            ''')
            conn.commit()

    def log_feedback(
        self,
        username: str,
        prompt: str,
        original_answer: str,
        rating: int, # 1 or -1
        image_hash: Optional[str] = None,
        user_correction: Optional[str] = None
    ):
        """Insert a piece of feedback into the RLHF store."""
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    '''INSERT INTO feedback 
                       (timestamp, username, image_hash, prompt, original_answer, user_correction, rating)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (timestamp, username, image_hash, prompt, original_answer, user_correction, rating)
                )
                conn.commit()
            logger.info("✅ Logged HITL Feedback to Flywheel.")
            
            # TODO: Add logic here to trigger Celery fine-tuning job if count > Threshold
            
        except Exception as e:
            logger.error(f"FLYWHEEL LOG FAILED: {e}")

_flywheel_instance = None

def get_flywheel() -> FlywheelDB:
    global _flywheel_instance
    if _flywheel_instance is None:
        _flywheel_instance = FlywheelDB()
    return _flywheel_instance

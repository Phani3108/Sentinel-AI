"""
Sentinel AI — Audit Logger
Provides immutable compliance logging for all queries.
"""
import sqlite3
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class AuditLogger:
    """Logs every prompt, user, and outcome to a local SQLite database for compliance auditing."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = "data/audit.db"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AuditLogger, cls).__new__(cls)
                    cls._instance._init(db_path)
        return cls._instance

    def _init(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the immutable table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    username TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    prompt TEXT,
                    image_hash TEXT,
                    route TEXT,
                    status_code INTEGER,
                    response_meta TEXT
                )
            ''')
            conn.commit()

    def log_request(
        self,
        username: str,
        endpoint: str,
        prompt: Optional[str] = None,
        image_hash: Optional[str] = None,
        route: str = "local",
        status_code: int = 200,
        response_meta: Optional[Dict[str, Any]] = None
    ):
        """Insert an immutable record of the interaction."""
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            meta_json = json.dumps(response_meta) if response_meta else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    '''INSERT INTO audit_log 
                       (timestamp, username, endpoint, prompt, image_hash, route, status_code, response_meta)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (timestamp, username, endpoint, prompt, image_hash, route, status_code, meta_json)
                )
                conn.commit()
        except Exception as e:
            # Audit failures must not crash the main application, but should be heavily logged.
            logger.error(f"AUDIT LOGGING FAILED: {e}")

    def query_logs(self, limit: int = 100) -> list:
        """Helper to fetch recent logs for admin review."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?', (limit,))
            return [dict(row) for row in cursor.fetchall()]

_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Singleton getter for the audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

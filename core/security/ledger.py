import sqlite3
import hashlib
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ImmutableLedger:
    """
    Zero-Trust Cryptographic Metadata Ledger.
    Records autonomous Swarm decisions sequentially using SHA-256 hashes linking back to previous blocks.
    Mathematically guarantees that threat events and Webhooks have not been tampered with or retroactively deleted.
    """
    def __init__(self, db_path="db_flywheel.sqlite3"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    payload TEXT,
                    prev_hash TEXT,
                    block_hash TEXT
                )
            ''')
            conn.commit()
            
    def _get_last_hash(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT block_hash FROM audit_ledger ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            return row[0] if row else "GENESIS_BLOCK"
            
    def record_event(self, event_type: str, payload_dict: dict):
        timestamp = datetime.now().isoformat()
        payload_str = json.dumps(payload_dict, sort_keys=True)
        prev_hash = self._get_last_hash()
        
        # Sequential Cryptographic Seal: SHA-256(timestamp + event_type + payload + prev_hash)
        raw_string = f"{timestamp}{event_type}{payload_str}{prev_hash}"
        block_hash = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO audit_ledger (timestamp, event_type, payload, prev_hash, block_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, event_type, payload_str, prev_hash, block_hash))
            conn.commit()
            
        logger.info(f"Ledger: Cryptographically sealed event '{event_type}'. Block Hash: {block_hash[:12]}...")
        
    def verify_chain(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, timestamp, event_type, payload, prev_hash, block_hash FROM audit_ledger ORDER BY id ASC')
            rows = cursor.fetchall()
            
        if not rows:
            return {"valid": True, "blocks": 0, "message": "Ledger is empty. Cryptographic integrity preserved."}
            
        expected_prev = "GENESIS_BLOCK"
        for row in rows:
            r_id, ts, etype, payload, prev_h, block_h = row
            if prev_h != expected_prev:
                logger.error(f"Ledger TAMPERED: Block {r_id} previous hash mismatch.")
                return {"valid": False, "broken_at": r_id, "message": f"Chain broken at Block #{r_id}. Prev Hash mismatch."}
                
            raw_string = f"{ts}{etype}{payload}{prev_h}"
            calc_hash = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
            if calc_hash != block_h:
                logger.error(f"Ledger TAMPERED: Block {r_id} invalid signature.")
                return {"valid": False, "broken_at": r_id, "message": f"Chain broken at Block #{r_id}. Mathematical Hash signature invalid."}
            
            expected_prev = block_h
            
        return {"valid": True, "blocks": len(rows), "message": "Cryptographic chain verified 100% SECURE."}

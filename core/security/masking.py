"""
Sentinel AI — Data Masking Layer
Physically redacts PII from prompts before they are sent to the LLM.
"""
import re
from typing import Tuple

class PIIMasker:
    """Uses regex rules to find and redact sensitive physical information."""
    
    # Highly specific regex patterns for US-centric PII
    PATTERNS = {
        "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "CREDIT_CARD": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
        "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "PHONE": re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b')
    }

    def __init__(self):
        # In an enterprise, we might initialize Microsoft Presidio here.
        pass

    def redact(self, text: str) -> Tuple[str, dict]:
        """
        Takes raw text and returns the redacted text alongside stats.
        Example: "My SSN is 123-45-6789" -> "My SSN is [REDACTED_SSN]"
        """
        if not text:
            return text, {}

        redacted_text = text
        stats = {key: 0 for key in self.PATTERNS.keys()}
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(redacted_text)
            if matches:
                stats[pii_type] += len(matches)
                redacted_text = pattern.sub(f"[REDACTED_{pii_type}]", redacted_text)
                
        return redacted_text, {k: v for k, v in stats.items() if v > 0}

_masker_instance = None

def get_pii_masker() -> PIIMasker:
    global _masker_instance
    if _masker_instance is None:
        _masker_instance = PIIMasker()
    return _masker_instance

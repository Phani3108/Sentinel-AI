"""
Sentinel AI — Sensitivity Classifier (Phase 6: Hybrid Router)
Detects PII, confidential content, and data classification labels.
"""
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Result from the sensitivity classifier."""
    label: str            # PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED
    score: float          # 0.0 - 1.0
    reasons: List[str] = field(default_factory=list)
    route: str = "local"  # local | cloud


# ── Keyword Banks ─────────────────────────────────────────────────────────────

_PII_KEYWORDS = [
    "social security", "ssn", "passport", "driving license", "driver's license",
    "date of birth", "dob", "credit card", "bank account", "iban", "routing number",
    "medical record", "diagnosis", "patient id", "health insurance",
    "biometric", "fingerprint", "retinal scan", "dna",
]

_INTERNAL_KEYWORDS = [
    "internal use only", "confidential", "proprietary", "trade secret",
    "not for distribution", "do not share", "restricted", "classified",
    "attorney-client", "privileged", "nda", "non-disclosure",
]

_FINANCIAL_KEYWORDS = [
    "quarterly revenue", "p&l", "profit and loss", "balance sheet",
    "earnings per share", "eps", "acquisition", "merger", "ipo",
    "insider", "material non-public",
]

# Regex patterns
_EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE    = re.compile(r"\b(\+?\d[\d\s\-().]{7,14}\d)\b")
_CREDIT_RE   = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN_RE      = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
_IP_RE       = re.compile(r"\b(?:192\.168|10\.|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b")


class SensitivityClassifier:
    """
    Rule-based sensitivity classifier that detects PII and confidential content
    in both text prompts and image metadata.

    Classification labels (in increasing sensitivity order):
    - PUBLIC: Safe to route to cloud
    - INTERNAL: May contain org-specific info, route locally by default
    - CONFIDENTIAL: Contains sensitive identifiers, always route locally
    - RESTRICTED: PII or regulated data, never leave local network
    """

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, anything INTERNAL+ routes locally (default: only CONFIDENTIAL+)
        """
        self.strict_mode = strict_mode

    def classify_text(self, text: str) -> ClassificationResult:
        """Classify a text prompt for sensitivity."""
        reasons = []
        text_lower = text.lower()
        score = 0.0

        # Check for regex patterns (high confidence PII)
        if _SSN_RE.search(text):
            reasons.append("Social Security Number detected")
            score = max(score, 0.95)

        if _CREDIT_RE.search(text):
            reasons.append("Potential credit card number detected")
            score = max(score, 0.90)

        if _EMAIL_RE.search(text):
            reasons.append("Email address detected")
            score = max(score, 0.60)

        if _PHONE_RE.search(text):
            reasons.append("Phone number detected")
            score = max(score, 0.55)

        if _IP_RE.search(text):
            reasons.append("Internal IP address detected")
            score = max(score, 0.50)

        # Check PII keywords
        for kw in _PII_KEYWORDS:
            if kw in text_lower:
                reasons.append(f"PII keyword: '{kw}'")
                score = max(score, 0.85)

        # Check confidential keywords
        for kw in _INTERNAL_KEYWORDS:
            if kw in text_lower:
                reasons.append(f"Confidential marker: '{kw}'")
                score = max(score, 0.75)

        # Check financial keywords
        for kw in _FINANCIAL_KEYWORDS:
            if kw in text_lower:
                reasons.append(f"Financial keyword: '{kw}'")
                score = max(score, 0.70)

        label = self._score_to_label(score)
        route = self._decide_route(label)

        return ClassificationResult(label=label, score=round(score, 3), reasons=reasons, route=route)

    def classify_image_path(self, image_path: str) -> ClassificationResult:
        """
        Heuristic classification based on image file path / metadata.
        (Real face-detection classification requires CV model; this is the rule-based fallback.)
        """
        reasons = []
        score = 0.0
        path_lower = str(image_path).lower()

        sensitive_dirs = ["medical", "patient", "private", "personal", "id", "passport",
                          "classified", "confidential", "restricted", "hr", "payroll"]
        for d in sensitive_dirs:
            if d in path_lower:
                reasons.append(f"Sensitive directory name: '{d}'")
                score = max(score, 0.70)

        label = self._score_to_label(score)
        route = self._decide_route(label)
        return ClassificationResult(label=label, score=round(score, 3), reasons=reasons, route=route)

    def classify(self, text: str = "", image_path: str = "") -> ClassificationResult:
        """Classify combining text + image path signals."""
        text_result  = self.classify_text(text) if text else ClassificationResult("PUBLIC", 0.0)
        image_result = self.classify_image_path(image_path) if image_path else ClassificationResult("PUBLIC", 0.0)

        # Take the maximum sensitivity
        combined_score = max(text_result.score, image_result.score)
        combined_reasons = list(set(text_result.reasons + image_result.reasons))
        label = self._score_to_label(combined_score)
        route = self._decide_route(label)

        return ClassificationResult(
            label=label,
            score=round(combined_score, 3),
            reasons=combined_reasons,
            route=route,
        )

    def _score_to_label(self, score: float) -> str:
        if score >= 0.85:
            return "RESTRICTED"
        elif score >= 0.65:
            return "CONFIDENTIAL"
        elif score >= 0.40:
            return "INTERNAL"
        else:
            return "PUBLIC"

    def _decide_route(self, label: str) -> str:
        """Decide routing based on sensitivity label."""
        if self.strict_mode:
            return "local" if label in ("INTERNAL", "CONFIDENTIAL", "RESTRICTED") else "cloud"
        return "local" if label in ("CONFIDENTIAL", "RESTRICTED") else "cloud"

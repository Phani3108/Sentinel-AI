import requests
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ActionEngine:
    """
    Physical bindings for Sentinel's autonomous remediation workflows.
    Executes actual HTTP REST operations mapped dynamically by the Action Agent node.
    """

    @staticmethod
    def dispatch_slack_alert(message: str, severity: str = "critical") -> Dict[str, Any]:
        """
        Dispatches a high-voltage Markdown payload to the enterprise Slack channel.
        For sandbox purposes, targets the internal FastAPI mock receiver.
        """
        webhook_url = "http://localhost:8080/system/mock/slack"
        
        payload = {
            "text": f"🚨 *SENTINEL AI: AUTONOMOUS REMEDIATION TRIGGERED*\n\n"
                    f"*Severity:* `{severity.upper()}`\n"
                    f"*Threat Manifest:* {message}\n"
        }

        try:
            logger.info(f"ActionEngine: Dispatching HTTP POST to Slack Webhook -> {webhook_url}")
            response = requests.post(
                webhook_url, 
                json=payload, 
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            response.raise_for_status()
            logger.info("ActionEngine: Webhook executed successfully.")
            return {"status": "success", "http_code": response.status_code}
        except Exception as e:
            logger.error(f"ActionEngine: Webhook dispatch failed -> {e}")
            return {"status": "error", "reason": str(e)}


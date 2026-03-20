"""
Sentinel AI — Hybrid Router (Phase 6)
Routes requests to local pipeline or cloud API based on sensitivity classification.
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Unified result from the hybrid router."""
    input_path: str
    prompt: str
    route: str              # "local" | "cloud"
    classification_label: str
    classification_score: float
    classification_reasons: list = field(default_factory=list)

    # Response
    vision_description: str = ""
    final_answer: str = ""
    model_used: str = ""

    # Performance
    total_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


class HybridRouter:
    """
    Routes image + prompt to either the local Sentinel pipeline
    or a cloud API (GPT-4V / Azure) based on content sensitivity.

    Decision logic:
    - RESTRICTED / CONFIDENTIAL → always local pipeline
    - PUBLIC / INTERNAL         → cloud (if available) or local fallback
    """

    def __init__(
        self,
        local_pipeline=None,
        cloud_client=None,
        strict_mode: bool = False,
        fallback_to_local: bool = True,
    ):
        """
        Args:
            local_pipeline: SentinelPipeline instance (or None to create on demand)
            cloud_client: CloudClient instance (or None if cloud is disabled)
            strict_mode: If True, INTERNAL content also routes locally
            fallback_to_local: If cloud fails, fall back to local pipeline
        """
        from router.sensitivity_classifier import SensitivityClassifier

        self._local_pipeline = local_pipeline
        self._cloud_client = cloud_client
        self.classifier = SensitivityClassifier(strict_mode=strict_mode)
        self.fallback_to_local = fallback_to_local

    def route(self, image_path: str, prompt: str) -> RouterResult:
        """
        Classify the request and route to the appropriate backend.

        Args:
            image_path: Path to the image file
            prompt: User question / instruction

        Returns:
            RouterResult with response, route decision, and cost info
        """
        t_start = time.perf_counter()

        # Step 1: Classify
        classification = self.classifier.classify(text=prompt, image_path=image_path)
        logger.info(
            f"[Router] Classified as {classification.label} (score={classification.score:.2f}) "
            f"→ route: {classification.route}"
        )

        result = RouterResult(
            input_path=image_path,
            prompt=prompt,
            route=classification.route,
            classification_label=classification.label,
            classification_score=classification.score,
            classification_reasons=classification.reasons,
        )

        # Step 2: Route
        if classification.route == "local":
            self._run_local(result)
        else:
            if self._cloud_client is not None:
                success = self._run_cloud(result)
                if not success and self.fallback_to_local:
                    logger.warning("[Router] Cloud failed — falling back to local pipeline")
                    result.route = "local (fallback)"
                    self._run_local(result)
            else:
                logger.info("[Router] No cloud client configured — routing locally")
                result.route = "local (no cloud configured)"
                self._run_local(result)

        result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        self._log_decision(result)
        return result

    def _run_local(self, result: RouterResult) -> None:
        """Run inference via the local SentinelPipeline."""
        try:
            pipeline = self._get_local_pipeline()
            pr = pipeline.run_image(result.input_path, prompt=result.prompt)
            result.vision_description = pr.vision_description
            result.final_answer = pr.final_answer
            result.model_used = f"{pr.vision_model} + {pr.llm_model}"
            result.tokens_used = pr.llm_tokens_used
            result.estimated_cost_usd = 0.0  # Local = free
        except Exception as e:
            logger.error(f"[Router] Local inference failed: {e}")
            result.success = False
            result.error = str(e)

    def _run_cloud(self, result: RouterResult) -> bool:
        """Run inference via the cloud client. Returns True on success."""
        try:
            cloud_result = self._cloud_client.analyze_image(
                image_path=result.input_path,
                prompt=result.prompt,
            )
            result.vision_description = cloud_result.get("description", "")
            result.final_answer = cloud_result.get("answer", "")
            result.model_used = cloud_result.get("model", "gpt-4o")
            result.tokens_used = cloud_result.get("tokens_used", 0)
            result.estimated_cost_usd = cloud_result.get("cost_usd", 0.0)
            return True
        except Exception as e:
            logger.warning(f"[Router] Cloud inference failed: {e}")
            return False

    def _get_local_pipeline(self):
        """Lazy-initialize the local pipeline."""
        if self._local_pipeline is None:
            from core.pipeline import SentinelPipeline
            self._local_pipeline = SentinelPipeline()
        return self._local_pipeline

    def _log_decision(self, result: RouterResult) -> None:
        """Log routing decision to the structured log file."""
        try:
            import json
            from datetime import datetime, timezone

            log_path = Path("data/routing_log.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "route": result.route,
                "classification": result.classification_label,
                "score": result.classification_score,
                "reasons": result.classification_reasons,
                "model_used": result.model_used,
                "latency_ms": round(result.total_latency_ms, 1),
                "cost_usd": result.estimated_cost_usd,
                "tokens_used": result.tokens_used,
                "success": result.success,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

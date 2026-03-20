"""
Sentinel AI — MLflow Tracking Helper (Phase 5)
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def start_run(experiment_name: str = "sentinel-ai-finetune", run_name: Optional[str] = None):
    """Start an MLflow run."""
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {run.info.run_id}")
        return run
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking")
        return None


def log_params(params: Dict[str, Any]):
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception:
        pass


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    try:
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
    except Exception:
        pass


def log_artifact(path: str):
    try:
        import mlflow
        mlflow.log_artifact(path)
    except Exception:
        pass


def end_run():
    try:
        import mlflow
        mlflow.end_run()
    except Exception:
        pass

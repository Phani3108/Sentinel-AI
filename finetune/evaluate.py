"""
Sentinel AI — Fine-Tuning Evaluation (Phase 5)
Evaluate a fine-tuned vision model on a held-out test set.
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating one VQA pair."""
    image_path: str
    question: str
    reference: str
    prediction: str
    bleu4: float = 0.0
    rouge_l: float = 0.0
    exact_match: bool = False


def compute_bleu4(reference: str, hypothesis: str) -> float:
    """Compute BLEU-4 score (simplified sentence BLEU)."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        sf = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25,)*4, smoothing_function=sf))
    except ImportError:
        # Fallback: unigram overlap
        ref_set = set(reference.lower().split())
        hyp_set = set(hypothesis.lower().split())
        if not hyp_set:
            return 0.0
        return len(ref_set & hyp_set) / len(hyp_set)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 (LCS-based)."""
    def _lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class ModelEvaluator:
    """
    Evaluate a fine-tuned vision model against a test dataset.

    Usage:
        evaluator = ModelEvaluator(model_path="finetune/checkpoints/florence2-lora")
        results = evaluator.evaluate_from_jsonl("finetune/data/florence_val.jsonl")
        report = evaluator.generate_report(results)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_pipeline: bool = False,
        api_url: str = "http://localhost:8080",
    ):
        """
        Args:
            model_path: Path to fine-tuned checkpoint (optional, uses API if None)
            use_pipeline: If True, load the model locally with HF Transformers
            api_url: Sentinel AI API URL (used when use_pipeline is False)
        """
        self.model_path = model_path
        self.use_pipeline = use_pipeline
        self.api_url = api_url
        self._model = None

    def _predict(self, image_path: str, question: str) -> str:
        """Generate a prediction for one image+question pair."""
        if self.use_pipeline and self.model_path:
            return self._predict_local(image_path, question)
        return self._predict_via_api(image_path, question)

    def _predict_via_api(self, image_path: str, question: str) -> str:
        """Use the Sentinel AI API for prediction."""
        try:
            import httpx
            with open(image_path, "rb") as f:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{self.api_url}/analyze/image",
                        files={"file": (Path(image_path).name, f, "image/jpeg")},
                        data={"prompt": question},
                    )
                resp.raise_for_status()
                return resp.json().get("final_answer", "")
        except Exception as e:
            logger.warning(f"API prediction failed: {e}")
            return ""

    def _predict_local(self, image_path: str, question: str) -> str:
        """Use a locally loaded HF model for prediction (requires GPU)."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            from PIL import Image
            import torch

            if self._model is None:
                logger.info(f"Loading model from {self.model_path}")
                self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, text=question, return_tensors="pt")
            outputs = self._model.generate(**inputs, max_new_tokens=256)
            return self._processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Local prediction error: {e}")
            return ""

    def evaluate_from_jsonl(self, jsonl_path: str, max_samples: Optional[int] = None) -> List[EvalResult]:
        """Evaluate all pairs in a JSONL file."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"{jsonl_path} not found")

        results = []
        with open(path) as f:
            lines = f.read().splitlines()

        if max_samples:
            lines = lines[:max_samples]

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                image_path = obj["image"]
                question = obj.get("question", "Describe this image.")
                reference = obj["answer"]

                logger.info(f"[{i+1}/{len(lines)}] Evaluating {Path(image_path).name}")
                prediction = self._predict(image_path, question)

                result = EvalResult(
                    image_path=image_path,
                    question=question,
                    reference=reference,
                    prediction=prediction,
                    bleu4=compute_bleu4(reference, prediction),
                    rouge_l=compute_rouge_l(reference, prediction),
                    exact_match=reference.strip().lower() == prediction.strip().lower(),
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate pair {i}: {e}")

        return results

    def generate_report(self, results: List[EvalResult]) -> dict:
        """Generate aggregate evaluation metrics."""
        if not results:
            return {"error": "No results to evaluate"}

        n = len(results)
        avg_bleu4 = sum(r.bleu4 for r in results) / n
        avg_rouge_l = sum(r.rouge_l for r in results) / n
        exact_match_rate = sum(1 for r in results if r.exact_match) / n

        report = {
            "num_samples": n,
            "avg_bleu4": round(avg_bleu4, 4),
            "avg_rouge_l": round(avg_rouge_l, 4),
            "exact_match_rate": round(exact_match_rate, 4),
            "model_path": self.model_path,
        }

        logger.info(f"Eval Report: {report}")
        return report

    def save_report(self, results: List[EvalResult], output_path: str = "finetune/reports/eval_report.json"):
        """Save detailed results + summary to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = self.generate_report(results)
        full_report = {
            "summary": summary,
            "details": [
                {
                    "image": r.image_path,
                    "question": r.question,
                    "reference": r.reference,
                    "prediction": r.prediction,
                    "bleu4": r.bleu4,
                    "rouge_l": r.rouge_l,
                    "exact_match": r.exact_match,
                }
                for r in results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Report saved → {output_path}")
        return full_report

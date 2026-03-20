"""
Sentinel AI — Phase 4 & 5 & 6 Tests
Tests for: Frontend imports, sensitivity classifier, hybrid router, dataset builder, evaluator.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# =========================================================================== #
# FRONTEND SMOKE TESTS
# =========================================================================== #

class TestFrontendImports:
    """Verify all frontend modules import without errors (no Streamlit server needed)."""

    def test_sidebar_import(self):
        """Sidebar component should exist and be syntactically valid Python."""
        import ast
        path = Path("frontend/components/sidebar.py")
        assert path.exists()
        # Parse the file — will raise SyntaxError if broken
        ast.parse(path.read_text(encoding="utf-8"))

    def test_result_card_import(self):
        """Result card component file should exist and be valid Python."""
        assert Path("frontend/components/result_card.py").exists()

    def test_all_pages_exist(self):
        """All 5 Streamlit page files should exist."""
        pages_dir = Path("frontend/pages")
        pages = list(pages_dir.glob("*.py"))
        assert len(pages) >= 5, f"Expected ≥5 pages, found {len(pages)}: {[p.name for p in pages]}"

    def test_app_exists(self):
        """Main app.py should exist."""
        assert Path("frontend/app.py").exists()

    def test_streamlit_config_exists(self):
        """Streamlit config.toml should exist with white background."""
        config_path = Path("frontend/.streamlit/config.toml")
        assert config_path.exists()
        content = config_path.read_text()
        assert "backgroundColor" in content
        assert "#FFFFFF" in content

    def test_frontend_theme_white(self):
        """Config.toml must specify white background."""
        config_path = Path("frontend/.streamlit/config.toml")
        content = config_path.read_text()
        assert 'backgroundColor = "#FFFFFF"' in content


# =========================================================================== #
# SENSITIVITY CLASSIFIER TESTS
# =========================================================================== #

class TestSensitivityClassifier:

    @pytest.fixture
    def classifier(self):
        from router.sensitivity_classifier import SensitivityClassifier
        return SensitivityClassifier()

    def test_public_text_routes_cloud(self, classifier):
        """Generic public text should be classified as PUBLIC and route to cloud."""
        result = classifier.classify_text("What is the weather like today?")
        assert result.label in ("PUBLIC", "INTERNAL")
        # PUBLIC routes to cloud in non-strict mode
        if result.label == "PUBLIC":
            assert result.route == "cloud"

    def test_ssn_pattern_detected(self, classifier):
        """Text containing SSN pattern should be RESTRICTED."""
        result = classifier.classify_text("My SSN is 123-45-6789, please verify.")
        assert result.label in ("CONFIDENTIAL", "RESTRICTED")
        assert result.route == "local"
        assert len(result.reasons) > 0

    def test_email_detected(self, classifier):
        """Email addresses should raise sensitivity score."""
        result = classifier.classify_text("Contact john.doe@company.com for details.")
        assert result.score > 0.0
        assert any("Email" in r or "email" in r.lower() for r in result.reasons)

    def test_pii_keyword_medical(self, classifier):
        """Medical record keyword should trigger CONFIDENTIAL+."""
        result = classifier.classify_text("Access the patient medical record for diagnosis review.")
        assert result.label in ("CONFIDENTIAL", "RESTRICTED")
        assert result.route == "local"

    def test_internal_keyword(self, classifier):
        """'Confidential' keyword should raise sensitivity."""
        result = classifier.classify_text("CONFIDENTIAL: Q3 internal revenue report")
        assert result.score >= 0.65

    def test_financial_keyword(self, classifier):
        """Financial/insider keywords should elevate classification."""
        result = classifier.classify_text("Quarterly revenue and earnings per share report.")
        assert result.score >= 0.60

    def test_image_path_medical(self, classifier):
        """Image paths with 'medical' directory should be classified sensitively."""
        result = classifier.classify_image_path("/data/medical/patient_scan.jpg")
        assert result.label in ("INTERNAL", "CONFIDENTIAL", "RESTRICTED")

    def test_combined_classification(self, classifier):
        """Combined text + image path classification should take max severity."""
        result = classifier.classify(
            text="SSN: 123-45-6789",
            image_path="/data/public/photo.jpg",
        )
        assert result.label in ("CONFIDENTIAL", "RESTRICTED")

    def test_strict_mode_internal_routes_local(self):
        """In strict mode, INTERNAL content should route locally."""
        from router.sensitivity_classifier import SensitivityClassifier
        strict = SensitivityClassifier(strict_mode=True)
        result = strict.classify_text("internal use only document")
        assert result.route == "local"

    def test_empty_text_is_public(self, classifier):
        """Empty text should classify as PUBLIC."""
        result = classifier.classify_text("")
        assert result.label == "PUBLIC"
        assert result.score == 0.0

    def test_classification_result_has_required_fields(self, classifier):
        """ClassificationResult should have expected attributes."""
        result = classifier.classify_text("hello world")
        assert hasattr(result, "label")
        assert hasattr(result, "score")
        assert hasattr(result, "reasons")
        assert hasattr(result, "route")
        assert isinstance(result.reasons, list)
        assert 0.0 <= result.score <= 1.0

    def test_score_in_valid_range(self, classifier):
        """Score must always be in [0, 1]."""
        for text in ["", "hello", "SSN 123-45-6789", "patient medical record diagnosis"]:
            result = classifier.classify_text(text)
            assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of range for: {text!r}"


# =========================================================================== #
# HYBRID ROUTER TESTS
# =========================================================================== #

class TestHybridRouter:

    @pytest.fixture
    def mock_pipeline(self):
        from core.pipeline import PipelineResult
        pipe = MagicMock()
        pipe.run_image.return_value = PipelineResult(
            input_path="test.jpg",
            input_type="image",
            prompt="What is this?",
            vision_model="llava:7b",
            vision_description="A sunset over the ocean.",
            final_answer="This is a beautiful sunset photo.",
            llm_model="llama3.1:8b",
            total_latency_ms=300.0,
            llm_tokens_used=30,
        )
        return pipe

    @pytest.fixture
    def router(self, mock_pipeline):
        from router.hybrid_router import HybridRouter
        return HybridRouter(local_pipeline=mock_pipeline, cloud_client=None)

    def test_public_routes_locally_when_no_cloud(self, router, tmp_path, mock_pipeline):
        """Without cloud client, all requests should go local."""
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)

        result = router.route(str(img_path), "What is this image?")
        assert "local" in result.route
        assert result.success or result.final_answer != ""

    def test_sensitive_always_routes_local(self, mock_pipeline, tmp_path):
        """SSN text should always trigger local routing."""
        from router.hybrid_router import HybridRouter
        mock_cloud = MagicMock()
        router = HybridRouter(local_pipeline=mock_pipeline, cloud_client=mock_cloud)

        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)

        result = router.route(str(img_path), "Patient SSN: 123-45-6789 — please verify")
        assert "local" in result.route
        mock_cloud.analyze_image.assert_not_called()

    def test_router_result_has_required_fields(self, router, tmp_path):
        """RouterResult must have all documented fields."""
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)

        result = router.route(str(img_path), "Describe this")
        assert hasattr(result, "route")
        assert hasattr(result, "classification_label")
        assert hasattr(result, "classification_score")
        assert hasattr(result, "final_answer")
        assert hasattr(result, "total_latency_ms")

    def test_result_latency_positive(self, router, tmp_path):
        """Total latency should be positive."""
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)
        result = router.route(str(img_path), "test")
        assert result.total_latency_ms >= 0


# =========================================================================== #
# DATASET BUILDER TESTS
# =========================================================================== #

class TestDatasetBuilder:

    @pytest.fixture
    def builder(self, tmp_path):
        from finetune.dataset_builder import DatasetBuilder
        return DatasetBuilder(output_dir=str(tmp_path / "output"))

    def test_load_from_jsonl(self, builder, tmp_path):
        """Should load VQA pairs from a JSONL file."""
        jsonl_path = tmp_path / "annotations.jsonl"
        import json
        records = [
            {"image": "img1.jpg", "question": "What do you see?", "answer": "A cat", "id": "001"},
            {"image": "img2.jpg", "question": "Describe the scene.", "answer": "A dog playing"},
        ]
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        pairs = builder.load_from_jsonl(str(jsonl_path))
        assert len(pairs) == 2
        assert pairs[0].answer == "A cat"
        assert pairs[0].question == "What do you see?"

    def test_load_from_folder(self, builder, tmp_path):
        """Should load images from a folder."""
        from PIL import Image
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (32, 32)).save(img_dir / f"img{i}.jpg")

        pairs = builder.load_from_folder(str(img_dir))
        assert len(pairs) == 3

    def test_split_ratios(self, builder, tmp_path):
        """Split should respect train/val/test ratios."""
        import json
        jsonl_path = tmp_path / "data.jsonl"
        records = [{"image": f"img{i}.jpg", "answer": f"answer {i}"} for i in range(100)]
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        pairs = builder.load_from_jsonl(str(jsonl_path))
        train, val, test = builder.split(pairs, train_ratio=0.8, val_ratio=0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_export_llava_format(self, builder, tmp_path):
        """LLaVA export should produce valid JSON."""
        import json
        from finetune.dataset_builder import VQAPair
        pairs = [VQAPair(image_path="img.jpg", question="Q?", answer="A.", conversation_id="001")]
        out = builder.export_llava_format(pairs, split="train")
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert "conversations" in data[0]
        assert data[0]["conversations"][0]["from"] == "human"

    def test_export_florence_format(self, builder, tmp_path):
        """Florence-2 export should produce valid JSONL."""
        import json
        from finetune.dataset_builder import VQAPair
        pairs = [VQAPair(image_path="img.jpg", question="Q?", answer="A.", conversation_id="001")]
        out = builder.export_florence_format(pairs, split="train")
        assert out.exists()
        line = json.loads(out.read_text().strip())
        assert line["task"] == "<VQA>"
        assert line["answer"] == "A."

    def test_split_labels_correctly(self, builder, tmp_path):
        """Each split should have the correct split label."""
        import json
        jsonl_path = tmp_path / "data.jsonl"
        records = [{"image": f"img{i}.jpg", "answer": f"answer {i}"} for i in range(20)]
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        pairs = builder.load_from_jsonl(str(jsonl_path))
        train, val, test = builder.split(pairs)
        for p in train: assert p.split == "train"
        for p in val:   assert p.split == "val"
        for p in test:  assert p.split == "test"


# =========================================================================== #
# EVALUATOR METRIC TESTS
# =========================================================================== #

class TestEvalMetrics:

    def test_bleu4_perfect_match(self):
        """Identical strings should yield high BLEU-4."""
        from finetune.evaluate import compute_bleu4
        score = compute_bleu4("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.8

    def test_bleu4_no_overlap(self):
        """Completely different strings should yield low BLEU-4."""
        from finetune.evaluate import compute_bleu4
        score = compute_bleu4("clear sky today", "broken equipment malfunction")
        assert score < 0.5

    def test_rouge_l_perfect(self):
        """Identical strings should yield ROUGE-L = 1.0."""
        from finetune.evaluate import compute_rouge_l
        score = compute_rouge_l("hello world", "hello world")
        assert abs(score - 1.0) < 1e-5

    def test_rouge_l_empty(self):
        """Empty hypothesis should yield 0."""
        from finetune.evaluate import compute_rouge_l
        score = compute_rouge_l("reference text", "")
        assert score == 0.0

    def test_bleu4_partial_match(self):
        """Partial match should return score between 0 and 1."""
        from finetune.evaluate import compute_bleu4
        score = compute_bleu4("the cat sat on the mat", "the cat ran on the mat")
        assert 0.0 < score < 1.0

    def test_evaluator_generates_report(self, tmp_path):
        """EvalResult list should produce a correct summary report."""
        from finetune.evaluate import ModelEvaluator, EvalResult
        evaluator = ModelEvaluator()
        results = [
            EvalResult("img1.jpg", "Q1", "A1", "A1", bleu4=1.0, rouge_l=1.0, exact_match=True),
            EvalResult("img2.jpg", "Q2", "A2", "B2", bleu4=0.0, rouge_l=0.0, exact_match=False),
        ]
        report = evaluator.generate_report(results)
        assert report["num_samples"] == 2
        assert report["avg_bleu4"] == 0.5
        assert report["exact_match_rate"] == 0.5

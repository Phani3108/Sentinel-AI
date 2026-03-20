"""
Sentinel AI — Dataset Builder (Phase 5: Fine-Tuning)
Converts image/caption annotations into training formats for LLaVA and Florence-2.
"""
import json
import logging
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VQAPair:
    """A single visual question-answering pair."""
    image_path: str
    question: str
    answer: str
    conversation_id: str = ""
    split: str = "train"   # train | val | test


class DatasetBuilder:
    """
    Build vision fine-tuning datasets from raw annotations.

    Supports output formats:
    - LLaVA instruction format (JSON)
    - Florence-2 VQA format (JSONL)
    - HuggingFace Dataset (via datasets library)
    """

    def __init__(self, output_dir: str = "finetune/data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Loading ─────────────────────────────────────────────────────────────────

    def load_from_jsonl(self, jsonl_path: str) -> List[VQAPair]:
        """Load VQA pairs from a JSONL file.

        Expected format per line:
            {"image": "path/to/img.jpg", "question": "...", "answer": "..."}
        """
        pairs = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {jsonl_path}")

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pairs.append(VQAPair(
                        image_path=obj["image"],
                        question=obj.get("question", "Describe this image."),
                        answer=obj["answer"],
                        conversation_id=obj.get("id", f"pair_{i:06d}"),
                    ))
                except (KeyError, json.JSONDecodeError) as e:
                    logger.warning(f"Skipping line {i}: {e}")

        logger.info(f"Loaded {len(pairs)} VQA pairs from {jsonl_path}")
        return pairs

    def load_from_folder(
        self,
        image_dir: str,
        caption_file: Optional[str] = None,
        default_question: str = "Describe this image in detail.",
    ) -> List[VQAPair]:
        """Load images from a folder. If a caption_file (.txt / .json) exists, load answers from it."""
        image_dir = Path(image_dir)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])

        captions: dict = {}
        if caption_file:
            cap_path = Path(caption_file)
            if cap_path.suffix == ".json":
                with open(cap_path) as f:
                    captions = json.load(f)
            elif cap_path.suffix == ".txt":
                with open(cap_path) as f:
                    lines = f.read().splitlines()
                    captions = {img.name: line for img, line in zip(images, lines)}

        pairs = []
        for img in images:
            answer = captions.get(img.name, captions.get(img.stem, "No caption provided."))
            pairs.append(VQAPair(
                image_path=str(img),
                question=default_question,
                answer=answer,
                conversation_id=img.stem,
            ))

        logger.info(f"Loaded {len(pairs)} pairs from {image_dir}")
        return pairs

    # ── Splitting ────────────────────────────────────────────────────────────────

    def split(
        self,
        pairs: List[VQAPair],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[VQAPair], List[VQAPair], List[VQAPair]]:
        """Split pairs into train/val/test sets."""
        random.seed(seed)
        shuffled = pairs.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = shuffled[:n_train]
        val = shuffled[n_train: n_train + n_val]
        test = shuffled[n_train + n_val:]

        for p in train: p.split = "train"
        for p in val:   p.split = "val"
        for p in test:  p.split = "test"

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    # ── Export ───────────────────────────────────────────────────────────────────

    def export_llava_format(self, pairs: List[VQAPair], split: str = "train") -> Path:
        """Export in LLaVA instruction tuning format."""
        records = []
        for p in pairs:
            records.append({
                "id": p.conversation_id,
                "image": p.image_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{p.question}"},
                    {"from": "gpt", "value": p.answer},
                ],
            })

        out_path = self.output_dir / f"llava_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(records)} LLaVA records → {out_path}")
        return out_path

    def export_florence_format(self, pairs: List[VQAPair], split: str = "train") -> Path:
        """Export in Florence-2 VQA JSONL format."""
        out_path = self.output_dir / f"florence_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for p in pairs:
                record = {
                    "task": "<VQA>",
                    "image": p.image_path,
                    "question": p.question,
                    "answer": p.answer,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(pairs)} Florence-2 records → {out_path}")
        return out_path

    def export_huggingface(self, pairs: List[VQAPair]):
        """Export as a HuggingFace Dataset (requires datasets library)."""
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            raise ImportError("Install `datasets`: pip install datasets")

        def _to_dict(p: VQAPair) -> dict:
            return asdict(p)

        train_pairs = [p for p in pairs if p.split == "train"]
        val_pairs   = [p for p in pairs if p.split == "val"]
        test_pairs  = [p for p in pairs if p.split == "test"]

        ds = DatasetDict({
            "train": Dataset.from_list([_to_dict(p) for p in train_pairs]),
            "val":   Dataset.from_list([_to_dict(p) for p in val_pairs]),
            "test":  Dataset.from_list([_to_dict(p) for p in test_pairs]),
        })

        out_path = self.output_dir / "hf_dataset"
        ds.save_to_disk(str(out_path))
        logger.info(f"HuggingFace dataset saved → {out_path}")
        return ds

    def build_all(self, jsonl_path: str) -> dict:
        """Full pipeline: load → split → export all formats."""
        pairs = self.load_from_jsonl(jsonl_path)
        train, val, test = self.split(pairs)
        all_pairs = train + val + test

        results = {
            "llava_train": str(self.export_llava_format(train, "train")),
            "llava_val":   str(self.export_llava_format(val, "val")),
            "florence_train": str(self.export_florence_format(train, "train")),
            "florence_val":   str(self.export_florence_format(val, "val")),
        }
        logger.info(f"Dataset build complete: {results}")
        return results

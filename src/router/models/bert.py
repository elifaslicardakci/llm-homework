"""Model 3: multilingual BERT fine-tuned for intent classification (PyTorch)."""
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from router.models.base import IntentModel

BASE_MODEL = "bert-base-multilingual-cased"
MAX_LEN = 32  # 95th percentile is 14–15 words; 32 word pieces cover almost every message
REPEAT_RE = re.compile(r"(\w)\1{2,}")


def preprocess(text: str) -> str:
    """Light normalization for BERT: keep case and punctuation, only fix Unicode and
    collapse long letter repeats (the cased tokenizer handles the rest)."""
    return REPEAT_RE.sub(r"\1\1", unicodedata.normalize("NFC", text)).strip()


class BertIntentModel(IntentModel):
    name = "mBERT"

    def __init__(self, model_dir: Path, threshold: float | None = None, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(device).eval()
        self.device = device
        cfg = self.model.config
        labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        super().__init__(labels, cfg.task_specific_params.get("threshold", 0.0)
                         if threshold is None and cfg.task_specific_params else threshold or 0.0)

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer([preprocess(t) for t in texts[i:i + batch_size]], padding=True,
                                 truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    def save_threshold(self) -> None:
        self.model.config.task_specific_params = {"threshold": self.threshold}
        self.model.config.save_pretrained(self.model_dir)

    def artifact_paths(self) -> list[Path]:
        return [p for p in self.model_dir.iterdir() if p.is_file()]

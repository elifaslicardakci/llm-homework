"""Distilled student exported to ONNX (FP32 or dynamic INT8), run with ONNX Runtime."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from router.models.base import IntentModel
from router.models.bert import MAX_LEN, preprocess


class OnnxIntentModel(IntentModel):
    def __init__(self, model_dir: Path, onnx_file: str = "model_int8.onnx", num_threads: int | None = None,
                 name: str = "mBERT student 4L · INT8 · ONNX"):
        self.model_dir = Path(model_dir)
        self.onnx_path = self.model_dir / onnx_file
        self.name = name
        cfg = json.loads((self.model_dir / "config.json").read_text())
        labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
        threshold = (cfg.get("task_specific_params") or {}).get("threshold", 0.0)
        super().__init__(labels, threshold)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer([preprocess(t) for t in texts[i:i + batch_size]], padding=True,
                                 truncation=True, max_length=MAX_LEN, return_tensors="np")
            feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            logits = self.session.run(["logits"], feeds)[0]
            e = np.exp(logits - logits.max(-1, keepdims=True))
            out.append(e / e.sum(-1, keepdims=True))
        return np.concatenate(out)

    def artifact_paths(self) -> list[Path]:
        tokenizer_files = [p for p in self.model_dir.iterdir()
                           if p.is_file() and p.suffix in {".json", ".txt"}]
        return [self.onnx_path, *tokenizer_files]

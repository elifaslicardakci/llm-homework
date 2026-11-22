"""Export the distilled student to ONNX, quantize it to dynamic INT8, evaluate both.

python scripts/quantize_bert.py [--skip-export]
"""
import argparse
import json

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from router.data import load_split
from router.evaluation.metrics import evaluate_model
from router.models.onnx_model import OnnxIntentModel

from distill_bert import STUDENT_DIR

FP32 = STUDENT_DIR / "model_fp32.onnx"
INT8 = STUDENT_DIR / "model_int8.onnx"


def export() -> None:
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_DIR).eval()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    enc = tokenizer(["bulaşık makinem su boşaltmıyor"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {n: {0: "batch", 1: "sequence"} for n in names}
    axes["logits"] = {0: "batch"}
    torch.onnx.export(model, tuple(enc[n] for n in names), FP32, input_names=names,
                      output_names=["logits"], dynamic_axes=axes, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(FP32))
    print(f"exported {FP32} ({FP32.stat().st_size / 1e6:.1f} MB)")


def quantize() -> None:
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8, per_channel=True)
    print(f"quantized {INT8} ({INT8.stat().st_size / 1e6:.1f} MB)")


def check_outputs() -> None:
    texts = load_split("val")["text"].tolist()[:256]
    torch_model = AutoModelForSequenceClassification.from_pretrained(STUDENT_DIR).eval()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    with torch.no_grad():
        enc = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        ref = torch.softmax(torch_model(**enc).logits, -1).numpy()
    for f in (FP32.name, INT8.name):
        probs = OnnxIntentModel(STUDENT_DIR, f).predict_proba(texts)
        agree = (probs.argmax(-1) == ref.argmax(-1)).mean()
        print(f"{f}: max |Δp| vs PyTorch {np.abs(probs - ref).max():.4f} · argmax agreement {agree:.4f}")


def evaluate() -> None:
    val, test = load_split("val"), load_split("test")
    for f, name in [(FP32.name, "mBERT student 4L · FP32 · ONNX"), (INT8.name, "mBERT student 4L · INT8 · ONNX")]:
        for threads, suffix in [(None, ""), (1, " · 1 thread")]:
            model = OnnxIntentModel(STUDENT_DIR, f, num_threads=threads, name=name + suffix)
            result = evaluate_model(model, val, test)
            print(json.dumps({"model": model.name, **{k: result[k] for k in ["threshold", "test", "size_mb", "latency"]}},
                             indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    if not args.skip_export:
        export()
        quantize()
    check_outputs()
    evaluate()

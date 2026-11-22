"""Collect reports/results/*.json into one results table (markdown + csv)."""
import json

import pandas as pd

from router.data import REPORTS_DIR

ORDER = [
    ("tf_idf_lr", "tf-idf + LR"),
    ("word2vec_ffn", "word2vec + FFN"),
    ("mbert", "mBERT teacher · FP32 · PyTorch"),
    ("mbert_student_4l", "mBERT student 4L · FP32 · PyTorch"),
    ("mbert_student_4l_fp32_onnx", "mBERT student 4L · FP32 · ONNX"),
    ("mbert_student_4l_int8_onnx", "mBERT student 4L · INT8 · ONNX (app)"),
]


def main() -> None:
    rows = []
    for slug, label in ORDER:
        path = REPORTS_DIR / "results" / f"{slug}.json"
        if not path.exists():
            print("skip (not evaluated):", slug)
            continue
        r = json.loads(path.read_text())
        one = REPORTS_DIR / "results" / f"{slug}_1_thread.json"
        lat1 = json.loads(one.read_text())["latency"]["mean_ms"] if one.exists() else None
        t = r["test"]
        rows.append({
            "Model": label,
            "Accuracy": round(t["accuracy"], 4),
            "In-scope acc.": round(t["in_scope_accuracy"], 4),
            "OOS recall": round(t["oos_recall"], 4),
            "Team acc.": round(t["team_accuracy"], 4),
            "Appliance acc.": round(t["appliance_accuracy"], 3),
            "Threshold": r["threshold"],
            "Size (MB)": round(r["size_mb"], 1),
            "Latency mean (ms)": round(r["latency"]["mean_ms"], 2),
            "Latency p95 (ms)": round(r["latency"]["p95_ms"], 2),
            "Latency 1 thread (ms)": None if lat1 is None else round(lat1, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "results_table.csv", index=False)
    hw = json.loads((REPORTS_DIR / "results" / f"{ORDER[0][0]}.json").read_text())["hardware"]
    md = (f"Test set: 5,524 messages (CLINC150 plus test incl. 1,000 out-of-scope + 24 hand-written "
          f"appliance messages). Latency: single-message calls, mean of 300, CPU "
          f"({hw['processor']}, Python {hw['python']}).\n\n" + df.to_markdown(index=False) + "\n")
    (REPORTS_DIR / "results_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

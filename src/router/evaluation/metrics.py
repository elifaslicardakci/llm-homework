"""Accuracy, out-of-scope recall, threshold tuning and latency for all models."""
import json
import platform
import time

import numpy as np
from sklearn.metrics import f1_score

from router.data import REPORTS_DIR
from router.models.base import IntentModel
from router.teams import OOS, team_of


def apply_threshold(probs: np.ndarray, labels: list[str], threshold: float) -> list[str]:
    return [OOS if p.max() < threshold else labels[int(p.argmax())] for p in probs]


def intent_metrics(gold: list[str], pred: list[str]) -> dict:
    gold, pred = np.asarray(gold), np.asarray(pred)
    in_scope, oos = gold != OOS, gold == OOS
    team_gold = np.array([team_of(g) for g in gold])
    team_pred = np.array([team_of(p) for p in pred])
    return {
        "accuracy": float((gold == pred).mean()),
        "in_scope_accuracy": float((gold[in_scope] == pred[in_scope]).mean()),
        "oos_recall": float((pred[oos] == OOS).mean()),
        "oos_precision": float((gold[pred == OOS] == OOS).mean()) if (pred == OOS).any() else 0.0,
        "macro_f1": float(f1_score(gold, pred, average="macro")),
        "team_accuracy": float((team_gold == team_pred).mean()),
    }


def tune_threshold(probs: np.ndarray, labels: list[str], gold: list[str]) -> tuple[float, dict]:
    """Pick the confidence threshold on validation data that maximizes the mean of
    in-scope accuracy and out-of-scope recall (validation has only 3% oos, so plain
    accuracy would almost ignore the out-of-scope class)."""
    best_t, best_score, best_m = 0.0, -1.0, {}
    for t in np.round(np.arange(0.0, 0.96, 0.01), 2):
        m = intent_metrics(gold, apply_threshold(probs, labels, t))
        score = (m["in_scope_accuracy"] + m["oos_recall"]) / 2
        if score > best_score:
            best_t, best_score, best_m = float(t), score, m
    return best_t, best_m


def hardware() -> dict:
    import torch
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
    }


def benchmark_latency(model: IntentModel, texts: list[str], n_runs: int = 300,
                      warmup: int = 10) -> dict:
    """Latency of single-message calls, as in the live app."""
    for t in texts[:warmup]:
        model.predict([t])
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        model.predict([texts[i % len(texts)]])
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "n_runs": n_runs,
    }


def evaluate_model(model: IntentModel, val, test, extra: dict | None = None) -> dict:
    """Tune the threshold on val, evaluate on test, measure size and latency, save JSON."""
    val_probs = model.predict_proba(val["text"].tolist())
    model.threshold, val_m = tune_threshold(val_probs, model.labels, val["intent"].tolist())
    test_probs = model.predict_proba(test["text"].tolist())
    test_pred = apply_threshold(test_probs, model.labels, model.threshold)
    test_m = intent_metrics(test["intent"].tolist(), test_pred)
    appliance = (test["source"] == "appliance").to_numpy()
    test_m["appliance_accuracy"] = float(
        (np.asarray(test_pred)[appliance] == test["intent"].to_numpy()[appliance]).mean()
    )
    result = {
        "model": model.name,
        "threshold": model.threshold,
        "val": val_m,
        "test": test_m,
        "size_mb": round(model.size_mb(), 2),
        "latency": benchmark_latency(model, test["text"].tolist()),
        "hardware": hardware(),
        **(extra or {}),
    }
    out = REPORTS_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    slug = model.name.lower().replace(" ", "_").replace("+", "").replace("__", "_")
    (out / f"{slug}.json").write_text(json.dumps(result, indent=2))
    return result

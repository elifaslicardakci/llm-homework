"""Train tf-idf + LR, pick C on validation, evaluate on test."""
import json

from router.data import MODELS_DIR, load_split
from router.evaluation.metrics import apply_threshold, evaluate_model, intent_metrics
from router.models.tfidf import TfidfModel


def main() -> None:
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    best_C, best_acc = None, -1.0
    for C in [3, 10, 30, 100]:
        model = TfidfModel.train(train["text"].tolist(), train["intent"].tolist(), C=C)
        pred = apply_threshold(model.predict_proba(val["text"].tolist()), model.labels, 0.0)
        acc = intent_metrics(val["intent"].tolist(), pred)["in_scope_accuracy"]
        print(f"C={C:<3} val in-scope accuracy {acc:.4f}")
        if acc > best_acc:
            best_C, best_acc = C, acc
    model = TfidfModel.train(train["text"].tolist(), train["intent"].tolist(), C=best_C)
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "tfidf_lr.joblib"
    model.save(path)
    result = evaluate_model(model, val, test, extra={"C": best_C})
    model.save(path)  # store the tuned threshold
    print(json.dumps({k: result[k] for k in ["threshold", "test", "size_mb", "latency"]}, indent=2))


if __name__ == "__main__":
    main()

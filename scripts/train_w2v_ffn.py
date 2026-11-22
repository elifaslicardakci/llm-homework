"""Train word2vec on the training messages, then the feedforward network on averaged vectors."""
import json

import numpy as np

from router.data import MODELS_DIR, label_list, load_split
from router.evaluation.metrics import evaluate_model
from router.models.w2v_ffn import W2VFFNModel, average_vectors, train_ffn, train_word2vec


def main() -> None:
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    labels = label_list()
    index = {label: i for i, label in enumerate(labels)}

    w2v = train_word2vec(train["text"].tolist())
    vectors = {w: w2v.wv[w] for w in w2v.wv.index_to_key}
    print(f"word2vec: {len(vectors)} words, dim {w2v.vector_size}")
    print("nearest to 'dishwasher':", [w for w, _ in w2v.wv.most_similar("dishwasher", topn=5)])

    x_train = average_vectors(train["text"].tolist(), vectors, w2v.vector_size)
    x_val = average_vectors(val["text"].tolist(), vectors, w2v.vector_size)
    y_train = train["intent"].map(index).to_numpy(np.int64)
    y_val = val["intent"].map(index).to_numpy(np.int64)
    net = train_ffn(x_train, y_train, x_val, y_val, n_classes=len(labels))

    model = W2VFFNModel(vectors, net, labels)
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "w2v_ffn.pt"
    model.save(path)
    result = evaluate_model(model, val, test, extra={"word2vec": {"dim": 100, "sg": 1, "window": 5,
                                                                   "vocab": len(vectors)}})
    model.save(path)  # store the tuned threshold
    print(json.dumps({k: result[k] for k in ["threshold", "test", "size_mb", "latency"]}, indent=2))


if __name__ == "__main__":
    main()

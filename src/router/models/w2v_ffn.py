"""Model 2: own word2vec (skip-gram) → averaged message vector → feedforward network,
trained with mini-batch SGD and backpropagation."""
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn

from router.models.base import IntentModel
from router.preprocessing.normalize import normalize, tokenize


def train_word2vec(texts: list[str], dim: int = 100, epochs: int = 30, seed: int = 42) -> Word2Vec:
    sentences = [tokenize(normalize(t)) for t in texts]
    return Word2Vec(sentences, vector_size=dim, window=5, min_count=1, sg=1,
                    negative=10, epochs=epochs, seed=seed, workers=1)


def average_vectors(texts: list[str], vectors: dict[str, np.ndarray] | object, dim: int) -> np.ndarray:
    """Mean of the word vectors of known tokens; zero vector if none is known."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        vecs = [vectors[w] for w in tokenize(normalize(text)) if w in vectors]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class W2VFFNModel(IntentModel):
    name = "word2vec + FFN"

    def __init__(self, vectors: dict[str, np.ndarray], net: FeedForward, labels: list[str],
                 threshold: float = 0.0, path: Path | None = None):
        super().__init__(labels, threshold)
        self.vectors = vectors
        self.dim = len(next(iter(vectors.values())))
        self.net = net.eval()
        self.path = path

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        x = torch.from_numpy(average_vectors(texts, self.vectors, self.dim))
        with torch.no_grad():
            return torch.softmax(self.net(x), dim=-1).numpy()

    def save(self, path: Path) -> None:
        words = list(self.vectors)
        torch.save({
            "words": words,
            "matrix": torch.from_numpy(np.stack([self.vectors[w] for w in words])),
            "state_dict": self.net.state_dict(),
            "hidden": self.net.net[0].out_features,
            "labels": self.labels,
            "threshold": self.threshold,
        }, path)
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "W2VFFNModel":
        obj = torch.load(path, weights_only=False)
        matrix = obj["matrix"].numpy()
        vectors = dict(zip(obj["words"], matrix))
        net = FeedForward(matrix.shape[1], obj["hidden"], len(obj["labels"]))
        net.load_state_dict(obj["state_dict"])
        return cls(vectors, net, obj["labels"], obj["threshold"], path)

    def artifact_paths(self) -> list[Path]:
        return [self.path]


def train_ffn(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
              n_classes: int, hidden: int = 256, lr: float = 0.1, momentum: float = 0.9,
              epochs: int = 60, batch_size: int = 64, seed: int = 42, log=print) -> FeedForward:
    """Mini-batch SGD with momentum; keeps the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    net = FeedForward(x_train.shape[1], hidden, n_classes)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(np.array(y_train))
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(np.array(y_val))
    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()          # backpropagation
            opt.step()               # SGD update
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            acc = (net(xv).argmax(-1) == yv).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            log(f"epoch {epoch:3d}  train loss {total / len(xt):.4f}  val acc {acc:.4f}")
    net.load_state_dict(best_state)
    log(f"best val acc {best_acc:.4f}")
    return net.eval()

"""Common interface for every intent model in the ladder."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from router.teams import OOS


class IntentModel(ABC):
    name: str = "base"

    def __init__(self, labels: list[str], threshold: float = 0.0):
        self.labels = list(labels)
        # Below this confidence the message goes to a human agent (predicted as oos).
        self.threshold = threshold

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """(n_texts, n_labels) class probabilities in the order of self.labels."""

    @abstractmethod
    def artifact_paths(self) -> list[Path]:
        """Files on disk the model needs at inference time (used for size)."""

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        """(intent, confidence) per text; low-confidence predictions become 'oos'."""
        probs = self.predict_proba(texts)
        return [
            (OOS if p.max() < self.threshold else self.labels[int(p.argmax())], float(p.max()))
            for p in probs
        ]

    def size_mb(self) -> float:
        return sum(p.stat().st_size for p in self.artifact_paths()) / 1e6

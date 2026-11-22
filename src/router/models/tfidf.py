"""Model 1: tf-idf (word + character n-grams) + logistic regression."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from router.models.base import IntentModel
from router.preprocessing.normalize import normalize


def build_pipeline(C: float = 10.0) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


class TfidfModel(IntentModel):
    name = "tf-idf + LR"

    def __init__(self, pipeline: Pipeline, threshold: float = 0.0, path: Path | None = None):
        super().__init__(list(pipeline.classes_), threshold)
        self.pipeline = pipeline
        self.path = path

    @classmethod
    def train(cls, texts: list[str], intents: list[str], C: float = 10.0) -> "TfidfModel":
        pipe = build_pipeline(C)
        pipe.fit([normalize(t) for t in texts], intents)
        return cls(pipe)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba([normalize(t) for t in texts])

    def save(self, path: Path) -> None:
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, path)
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "TfidfModel":
        obj = joblib.load(path)
        return cls(obj["pipeline"], obj["threshold"], path)

    def artifact_paths(self) -> list[Path]:
        return [self.path]

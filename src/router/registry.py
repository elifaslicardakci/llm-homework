"""Load every trained model that is available on disk, in ladder order."""
from collections.abc import Callable
from pathlib import Path

from router.data import MODELS_DIR
from router.models.base import IntentModel

DEFAULT_MODEL = "mBERT student 4L · INT8 · ONNX"


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))


def _tfidf() -> IntentModel:
    from router.models.tfidf import TfidfModel
    _require(MODELS_DIR / "tfidf_lr.joblib")
    return TfidfModel.load(MODELS_DIR / "tfidf_lr.joblib")


def _w2v() -> IntentModel:
    from router.models.w2v_ffn import W2VFFNModel
    _require(MODELS_DIR / "w2v_ffn.pt")
    return W2VFFNModel.load(MODELS_DIR / "w2v_ffn.pt")


def _teacher() -> IntentModel:
    from router.models.bert import BertIntentModel
    _require(MODELS_DIR / "bert_teacher" / "config.json")
    model = BertIntentModel(MODELS_DIR / "bert_teacher")
    model.name = "mBERT teacher · FP32 · PyTorch"
    return model


def _student_int8() -> IntentModel:
    from router.models.onnx_model import OnnxIntentModel
    _require(MODELS_DIR / "bert_student" / "model_int8.onnx")
    return OnnxIntentModel(MODELS_DIR / "bert_student", "model_int8.onnx", name=DEFAULT_MODEL)


# Default (quantized student) first: the app selects it on start.
MODEL_LOADERS: dict[str, Callable[[], IntentModel]] = {
    DEFAULT_MODEL: _student_int8,
    "mBERT teacher · FP32 · PyTorch": _teacher,
    "word2vec + FFN": _w2v,
    "tf-idf + LR": _tfidf,
}


def load_available() -> tuple[dict[str, IntentModel], dict[str, str]]:
    """Return (loaded models by name, missing models with the reason)."""
    models, missing = {}, {}
    for name, loader in MODEL_LOADERS.items():
        try:
            models[name] = loader()
        except FileNotFoundError as e:
            missing[name] = f"missing file: {e}"
    return models, missing

"""Download the trained models from the Hugging Face Hub into models/.

python scripts/download_models.py [--repo USER/REPO] [--app-only]
--app-only skips the 700 MB teacher; the app then runs the other three models.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "elifaslicardakci/llm-homework"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--app-only", action="store_true")
    args = parser.parse_args()
    ignore = ["bert_teacher/*", "bert_student/model.safetensors", "bert_student/model_fp32.onnx"] if args.app_only else None
    path = snapshot_download(args.repo, local_dir=MODELS_DIR, ignore_patterns=ignore)
    print("models in", path)


if __name__ == "__main__":
    main()

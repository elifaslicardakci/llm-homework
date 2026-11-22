"""Upload models/ to a Hugging Face model repo, one file at a time with retries.

Log in first: hf auth login
python scripts/upload_models.py [--repo USER/REPO]
"""
import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi

from download_models import DEFAULT_REPO, MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    existing = set(api.list_repo_files(args.repo))
    files = [p for p in MODELS_DIR.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    for path in sorted(files):
        rel = path.relative_to(MODELS_DIR).as_posix()
        if rel in existing:
            print("skip", rel)
            continue
        for attempt in range(1, 4):
            try:
                api.upload_file(path_or_fileobj=str(path), path_in_repo=rel, repo_id=args.repo)
                print(f"uploaded {rel} ({path.stat().st_size / 1e6:.1f} MB)")
                break
            except Exception as e:  # network errors: retry
                print(f"attempt {attempt} failed for {rel}: {e}")
                time.sleep(5 * attempt)
        else:
            raise SystemExit(f"could not upload {rel}")


if __name__ == "__main__":
    main()

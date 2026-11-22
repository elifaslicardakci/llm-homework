"""Fine-tune multilingual BERT for intent classification; keep the best epoch on validation.

python scripts/train_bert.py [--epochs 3] [--lr 5e-5] [--batch-size 32] [--eval-only]
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from router.data import MODELS_DIR, label_list, load_split
from router.evaluation.metrics import evaluate_model
from router.models.bert import BASE_MODEL, MAX_LEN, BertIntentModel, preprocess

OUT_DIR = MODELS_DIR / "bert_teacher"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def batches(df, tokenizer, index, batch_size, shuffle):
    rows = list(zip(df["text"].map(preprocess), df["intent"].map(index)))

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def accuracy(model, loader, dev) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(dev)
        pred = model(**{k: v for k, v in batch.items() if k != "labels"}).logits.argmax(-1)
        correct += (pred == batch["labels"]).sum().item()
        total += len(pred)
    return correct / total


def train(args) -> None:
    torch.manual_seed(42)
    train_df, val_df = load_split("train"), load_split("val")
    labels = label_list()
    index = {label: i for i, label in enumerate(labels)}
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=index,
    ).to(dev)
    train_loader = batches(train_df, tokenizer, index, args.batch_size, shuffle=True)
    val_loader = batches(val_df, tokenizer, index, 128, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    print(f"device {dev} · {len(train_df)} train messages · {len(labels)} intents · {steps} steps", flush=True)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            loss = model(**batch.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {total / step:.4f}", flush=True)
        acc = accuracy(model, val_loader, dev)
        print(f"epoch {epoch} done in {time.time() - start:.0f}s · train loss {total / len(train_loader):.4f}"
              f" · val accuracy {acc:.4f}", flush=True)
        if acc > best:
            best = acc
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            print(f"saved best model to {OUT_DIR}", flush=True)
        if dev == "mps":
            torch.mps.empty_cache()


def evaluate() -> None:
    model = BertIntentModel(OUT_DIR)
    result = evaluate_model(model, load_split("val"), load_split("test"),
                            extra={"base_model": BASE_MODEL, "max_len": MAX_LEN})
    model.save_threshold()
    print(json.dumps({k: result[k] for k in ["threshold", "test", "size_mb", "latency"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if not args.eval_only:
        train(args)
    evaluate()

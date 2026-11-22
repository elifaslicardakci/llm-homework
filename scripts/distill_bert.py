"""Knowledge distillation: 12-layer mBERT teacher → 4-layer student.

The student copies the teacher's embeddings and layers 0, 4, 8, 11, then is trained on
loss = α · KL(student_T ‖ teacher_T) · T² + (1 − α) · cross-entropy(gold).

python scripts/distill_bert.py [--epochs 4] [--temperature 2] [--alpha 0.5] [--eval-only]
"""
import argparse
import copy
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from router.data import MODELS_DIR, label_list, load_split
from router.evaluation.metrics import evaluate_model
from router.models.bert import BertIntentModel

from train_bert import OUT_DIR as TEACHER_DIR, accuracy, batches, device

STUDENT_DIR = MODELS_DIR / "bert_student"
KEEP_LAYERS = [0, 4, 8, 11]


def make_student(teacher):
    student = copy.deepcopy(teacher)
    layers = student.bert.encoder.layer
    student.bert.encoder.layer = torch.nn.ModuleList([layers[i] for i in KEEP_LAYERS])
    student.config.num_hidden_layers = len(KEEP_LAYERS)
    return student


def train(args) -> None:
    torch.manual_seed(42)
    dev = device()
    labels = label_list()
    index = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_DIR)
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_DIR).to(dev).eval()
    student = make_student(teacher).train()
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    print(f"teacher {n_t / 1e6:.1f}M params · student {n_s / 1e6:.1f}M params", flush=True)

    train_loader = batches(load_split("train"), tokenizer, index, args.batch_size, shuffle=True)
    val_loader = batches(load_split("val"), tokenizer, index, 128, shuffle=False)
    print(f"teacher val accuracy {accuracy(teacher, val_loader, dev):.4f}", flush=True)
    print(f"student val accuracy before distillation {accuracy(student, val_loader, dev):.4f}", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    T, alpha = args.temperature, args.alpha
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        start, total = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = batch.to(dev)
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                t_logits = teacher(**inputs).logits
            s_logits = student(**inputs).logits
            soft = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1),
                            reduction="batchmean") * T * T
            hard = F.cross_entropy(s_logits, batch["labels"])
            loss = alpha * soft + (1 - alpha) * hard
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {total / step:.4f}", flush=True)
        acc = accuracy(student, val_loader, dev)
        print(f"epoch {epoch} done in {time.time() - start:.0f}s · loss {total / len(train_loader):.4f}"
              f" · val accuracy {acc:.4f}", flush=True)
        if acc > best:
            best = acc
            student.save_pretrained(STUDENT_DIR)
            tokenizer.save_pretrained(STUDENT_DIR)
            print(f"saved best student to {STUDENT_DIR}", flush=True)
        if dev == "mps":
            torch.mps.empty_cache()


def evaluate(args) -> None:
    model = BertIntentModel(STUDENT_DIR)
    model.name = "mBERT student 4L"
    result = evaluate_model(model, load_split("val"), load_split("test"), extra={
        "teacher": str(TEACHER_DIR.name), "kept_layers": KEEP_LAYERS,
        "temperature": args.temperature, "alpha": args.alpha,
    })
    model.save_threshold()
    print(json.dumps({k: result[k] for k in ["threshold", "test", "size_mb", "latency"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if not args.eval_only:
        train(args)
    evaluate(args)

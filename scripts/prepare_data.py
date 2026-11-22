"""Download CLINC150 (plus), add the hand-written appliance examples, write the splits.

CLINC150 keeps its official train / validation / test splits. The 100 appliance
examples (25 per intent, Turkish and English) are split per intent and language
into 15 train / 4 validation / 6 test. Template-generated appliance messages
(router.augment) are added to the training split only.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

from router.augment import augment
from router.preprocessing.normalize import KNOWN_WORDS_FILE, normalize
from router.teams import OOS, RESOURCES, domain_of, team_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
SEED = 42
DOMAIN_SPLIT = {"train": 0.6, "val": 0.16}  # rest → test


def load_clinc() -> dict[str, pd.DataFrame]:
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    splits = {}
    for hf_name, name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        df = ds[hf_name].to_pandas()
        df["intent"] = df["intent"].map(lambda i: names[i])
        df["lang"] = "en"
        df["source"] = "clinc150"
        splits[name] = df
    return splits


def split_domain_examples() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(RESOURCES / "domain_examples.csv")
    df["source"] = "appliance"
    parts = {"train": [], "val": [], "test": []}
    for _, group in df.groupby(["intent", "lang"]):
        group = group.sample(frac=1, random_state=SEED)
        n_train = round(len(group) * DOMAIN_SPLIT["train"])
        n_val = round(len(group) * DOMAIN_SPLIT["val"])
        parts["train"].append(group.iloc[:n_train])
        parts["val"].append(group.iloc[n_train:n_train + n_val])
        parts["test"].append(group.iloc[n_train + n_val:])
    return {k: pd.concat(v) for k, v in parts.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clinc, domain = load_clinc(), split_domain_examples()
    stats = {}
    for name in ["train", "val", "test"]:
        parts = [clinc[name], domain[name]]
        if name == "train":
            parts.append(pd.DataFrame(augment()))
        df = pd.concat(parts, ignore_index=True)
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        df["domain"] = df["intent"].map(domain_of)
        df["team"] = df["intent"].map(team_of)
        df[["text", "intent", "domain", "team", "lang", "source"]].to_parquet(OUT / f"{name}.parquet")
        n_words = df["text"].str.split().str.len()
        stats[name] = {
            "messages": len(df),
            "intents": int(df["intent"].nunique()),
            "oos": int((df["intent"] == OOS).sum()),
            "appliance": int((df["source"] == "appliance").sum()),
            "appliance_augmented": int((df["source"] == "appliance_aug").sum()),
            "turkish": int((df["lang"] == "tr").sum()),
            "mean_words": round(float(n_words.mean()), 1),
            "p95_words": int(n_words.quantile(0.95)),
        }
        print(name, stats[name])
        if name == "train":
            write_known_words(df["text"])
    (ROOT / "reports" / "data_stats.json").write_text(json.dumps(stats, indent=2))
    plot(stats)


def write_known_words(texts: pd.Series) -> None:
    """Training vocabulary: the product-name corrector never changes these words."""
    words = {w for t in texts for w in normalize(t, correct=False).split()}
    KNOWN_WORDS_FILE.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    print("known words:", len(words))


def plot(stats: dict) -> None:
    names = list(stats)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    appliance = [stats[n]["appliance"] + stats[n]["appliance_augmented"] for n in names]
    oos = [stats[n]["oos"] for n in names]
    in_scope = [stats[n]["messages"] - o - a for n, o, a in zip(names, oos, appliance)]
    ax.bar(names, in_scope, label="CLINC150 in-scope", color="#4C72B0")
    ax.bar(names, oos, bottom=in_scope, label="Out-of-scope", color="#DD8452")
    bottom = [a + b for a, b in zip(in_scope, oos)]
    ax.bar(names, appliance, bottom=bottom, label="Appliance (TR + EN, train incl. templates)", color="#55A868")
    ax.set_ylabel("Messages")
    ax.set_title("Data splits")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "data_overview.png", dpi=160)


if __name__ == "__main__":
    main()

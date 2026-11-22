"""Architecture diagram and technique map (reports/figures)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
BLUE, GREEN, ORANGE, GREY, INK = "#DCE8FA", "#DDF3E4", "#FDE7D3", "#EEF0F3", "#1F2937"


def box(ax, x, y, w, h, text, color, size=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#98A2B3", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color="#475467", lw=1.3))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=8, color="#475467")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(0.1, 5.9, "Online: one Gradio process on CPU (app/app.py → router/pipeline.py)", fontsize=12,
            fontweight="bold", color=INK)
    box(ax, 0.1, 3.6, 1.8, 1.2, "Customer message\n(Gradio textbox)", GREY)
    box(ax, 2.4, 3.6, 2.2, 1.2, "Preprocessing\nmodel-code regex\nnormalize + edit-distance\nproduct-name fix", BLUE, 9)
    box(ax, 5.1, 4.6, 2.0, 0.9, "ELIZA rules\n(greeting, thanks …)", ORANGE, 9)
    box(ax, 5.1, 2.3, 2.6, 2.0, "Intent model (dropdown)\n• mBERT student 4L\n  INT8 · ONNX Runtime (default)\n"
        "• mBERT teacher FP32\n• word2vec + FFN\n• tf-idf + LR", GREEN, 9)
    box(ax, 8.3, 3.0, 2.0, 1.2, "Confidence ≥ threshold?\n(tuned on validation)", GREY, 9)
    box(ax, 10.9, 4.6, 2.0, 0.9, "Auto reply\n(small talk)", ORANGE, 9)
    box(ax, 10.9, 3.3, 2.0, 0.9, "Team\n(intent → domain → team)", BLUE, 9)
    box(ax, 10.9, 2.0, 2.0, 0.9, "Human agent\n(oos / low confidence)", ORANGE, 9)
    box(ax, 13.3, 2.4, 1.6, 2.4, "UI output\nteam, intent,\nconfidence,\nmodel codes,\nlatency +\nall-models table", GREY, 9)
    arrow(ax, 1.9, 4.2, 2.4, 4.2)
    arrow(ax, 4.6, 4.5, 5.1, 5.0, "rule match")
    arrow(ax, 4.6, 3.9, 5.1, 3.5, "no match")
    arrow(ax, 7.7, 3.5, 8.3, 3.6, "P(intent)")
    arrow(ax, 10.3, 3.8, 10.9, 3.75, "yes")
    arrow(ax, 10.3, 3.3, 10.9, 2.45, "no")
    arrow(ax, 7.1, 5.05, 10.9, 5.05)
    for y in (5.05, 3.75, 2.45):
        arrow(ax, 12.9, y, 13.3, 3.6)

    ax.plot([0.1, 14.9], [1.15, 1.15], ls="--", color="#D0D5DD")
    ax.text(0.1, 0.85, "Offline (scripts/)", fontsize=12, fontweight="bold", color=INK)
    steps = ["prepare_data.py\nCLINC150 + appliance", "train_tfidf.py", "train_w2v_ffn.py", "train_bert.py\nmBERT teacher",
             "distill_bert.py\n4-layer student", "quantize_bert.py\nONNX + INT8", "benchmark.py\nresults table"]
    w = 1.85
    for i, s in enumerate(steps):
        x = 0.1 + i * (w + 0.25)
        box(ax, x, 0.05, w, 0.7, s, GREY, 8)
        if i:
            arrow(ax, x - 0.25, 0.4, x, 0.4)
    fig.tight_layout()
    fig.savefig(FIG / "architecture.png", dpi=160)


def technique_map() -> None:
    areas = [
        ("A · Neural networks from scratch", "Own word2vec (skip-gram) on training messages →\n"
         "averaged vectors → feedforward net, SGD + backprop", GREEN),
        ("B · Classical NLP", "Regex normalization + model-code extraction ·\nedit-distance product-name correction ·\n"
         "ELIZA small-talk rules · tf-idf + logistic regression", BLUE),
        ("C · Transformers", "Multilingual BERT fine-tuned for 155 intents\n(Turkish + English)", ORANGE),
        ("D · Efficiency", "Knowledge distillation 12 → 4 layers ·\nONNX export + dynamic INT8 quantization ·\n"
         "latency per message", GREY),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
    for i, (title, body, color) in enumerate(areas):
        x, y = (i % 2) * 6 + 0.1, 2.7 - (i // 2) * 2.6
        ax.add_patch(FancyBboxPatch((x, y), 5.7, 2.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=color, ec="#98A2B3"))
        ax.text(x + 0.25, y + 1.9, title, fontsize=13, fontweight="bold", color=INK)
        ax.text(x + 0.25, y + 1.45, body, fontsize=10.5, color=INK, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "technique_map.png", dpi=160)


if __name__ == "__main__":
    architecture()
    technique_map()

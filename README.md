# Customer Support Ticket Router (SWE015 · Use Case 2)

Routes Turkish and English support messages for a home-appliance brand to the right team,
with a confidence score. Out-of-scope or low-confidence messages go to a human agent;
greetings and small talk get a rule-based (ELIZA-style) reply. The app shows the
predictions of all four models side by side.

Elif Aslı Çardakcı · 2309111004

## Model ladder

| # | Model | Area |
|---|---|---|
| 1 | tf-idf (word 1–2 + char 2–5 grams) + logistic regression | B |
| 2 | Own word2vec (skip-gram, trained on the training messages) → averaged vectors → feedforward net, SGD + backprop | A |
| 3 | `bert-base-multilingual-cased` fine-tuned for 155 intents (teacher) | C |
| 4 | 4-layer student distilled from the teacher → ONNX → dynamic INT8 (default in the app) | D |

Preprocessing (area B): regex normalization (URLs, e-mails, phones, model codes such as `DW-6534`),
Turkish-aware lowercasing, edit-distance correction of misspelled product names
(`bulasik makinasi` → `bulaşık makinesi`), ELIZA-style small-talk rules.

## Data

- [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) "plus": 150 intents in 10 domains + out-of-scope,
  official train / validation / test splits (test is used for every model).
- 100 hand-written appliance messages (Turkish and English, 4 intents: fault, warranty, spare parts,
  installation) in `src/router/resources/domain_examples.csv`, split 15 / 4 / 6 per intent.
- 200 template-generated appliance messages, **training split only** (`src/router/augment.py`).
- Routing: intent → CLINC domain or appliance intent → team (`src/router/teams.py`).

## Setup

Python 3.10+ (tested on 3.13, macOS, Apple M2, CPU).

```bash
git clone https://github.com/elifaslicardakci/llm-homework
cd llm-homework
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py        # trained models from the Hugging Face Hub
python app/app.py                        # http://127.0.0.1:7860
```

`python scripts/download_models.py --app-only` skips the 700 MB teacher model.

## Reproduce

```bash
export PYTHONPATH=src
python scripts/prepare_data.py      # data/processed/*.parquet, reports/data_stats.json
python scripts/train_tfidf.py
python scripts/train_w2v_ffn.py
python scripts/train_bert.py        # ~15 min on Apple MPS
python scripts/distill_bert.py
python scripts/quantize_bert.py     # ONNX export, INT8, evaluation
python scripts/benchmark.py         # reports/results_table.md
python scripts/make_figures.py
pytest
```

The decision threshold of every model is tuned on validation data to maximize the mean of
in-scope accuracy and out-of-scope recall. Results per model: `reports/results/*.json`.

## Results

Test set: 5,524 messages (CLINC150 plus test incl. 1,000 out-of-scope + 24 hand-written appliance messages). Latency: single-message calls, mean of 300, Apple M2 CPU (macOS arm64, Python 3.13.0).

| Model                                |   Accuracy |   In-scope acc. |   OOS recall |   Team acc. |   Appliance acc. |   Threshold |   Size (MB) |   Latency mean (ms) |   Latency p95 (ms) | Latency 1 thread (ms)   |
|:-------------------------------------|-----------:|----------------:|-------------:|------------:|-----------------:|------------:|------------:|--------------------:|-------------------:|:------------------------|
| tf-idf + LR                          |     0.8675 |          0.8603 |        0.9   |      0.8849 |            0.5   |        0.56 |        67.6 |                0.59 |               0.74 | –                       |
| word2vec + FFN                       |     0.8186 |          0.8112 |        0.852 |      0.8421 |            0.208 |        0.79 |         2.5 |                0.05 |               0.06 | –                       |
| mBERT teacher · FP32 · PyTorch       |     0.8463 |          0.8382 |        0.883 |      0.8612 |            0     |        0.6  |       714.8 |               21.74 |              22.76 | –                       |
| mBERT student 4L · FP32 · PyTorch    |     0.8698 |          0.8652 |        0.891 |      0.882  |            0.458 |        0.72 |       488   |               11.22 |              15.59 | –                       |
| mBERT student 4L · FP32 · ONNX       |     0.8698 |          0.8652 |        0.891 |      0.882  |            0.458 |        0.72 |       488.1 |                7.68 |              11.18 | 8.96                    |
| mBERT student 4L · INT8 · ONNX (app) |     0.8644 |          0.8572 |        0.897 |      0.8753 |            0.458 |        0.74 |       124.6 |                1.76 |               2.57 | 3.0                     |

The teacher was trained for 2 epochs: the epoch-3 checkpoint (validation accuracy 0.932) could not be saved because the disk was full. The distilled student had 3 more epochs and ends up more accurate than the teacher.

## Repository layout

```
app/app.py                  Gradio app
src/router/preprocessing/   normalization, model-code regex, product-name correction
src/router/eliza.py         small-talk rules
src/router/models/          tf-idf, word2vec FFN, BERT, ONNX models
src/router/pipeline.py      message → rules or model → threshold → team
scripts/                    data, training, distillation, quantization, benchmark, figures
reports/                    results, logs, figures
tests/                      pytest
```

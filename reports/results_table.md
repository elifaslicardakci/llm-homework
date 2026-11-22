Test set: 5,524 messages (CLINC150 plus test incl. 1,000 out-of-scope + 24 hand-written appliance messages). Latency: single-message calls, mean of 300, Apple M2 CPU (macOS arm64, Python 3.13.0).

| Model                                |   Accuracy |   In-scope acc. |   OOS recall |   Team acc. |   Appliance acc. |   Threshold |   Size (MB) |   Latency mean (ms) |   Latency p95 (ms) | Latency 1 thread (ms)   |
|:-------------------------------------|-----------:|----------------:|-------------:|------------:|-----------------:|------------:|------------:|--------------------:|-------------------:|:------------------------|
| tf-idf + LR                          |     0.8675 |          0.8603 |        0.9   |      0.8849 |            0.5   |        0.56 |        67.6 |                0.59 |               0.74 | –                       |
| word2vec + FFN                       |     0.8186 |          0.8112 |        0.852 |      0.8421 |            0.208 |        0.79 |         2.5 |                0.05 |               0.06 | –                       |
| mBERT teacher · FP32 · PyTorch       |     0.8463 |          0.8382 |        0.883 |      0.8612 |            0     |        0.6  |       714.8 |               21.74 |              22.76 | –                       |
| mBERT student 4L · FP32 · PyTorch    |     0.8698 |          0.8652 |        0.891 |      0.882  |            0.458 |        0.72 |       488   |               11.22 |              15.59 | –                       |
| mBERT student 4L · FP32 · ONNX       |     0.8698 |          0.8652 |        0.891 |      0.882  |            0.458 |        0.72 |       488.1 |                7.68 |              11.18 | 8.96                    |
| mBERT student 4L · INT8 · ONNX (app) |     0.8644 |          0.8572 |        0.897 |      0.8753 |            0.458 |        0.74 |       124.6 |                1.76 |               2.57 | 3.0                     |

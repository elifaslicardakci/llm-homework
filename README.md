# Customer Support Ticket Router (SWE015 · Use Case 2)

Routes Turkish and English support messages to the right team, with a confidence score,
an out-of-scope fallback to a human agent and rule-based small-talk replies.

Model ladder: tf-idf + logistic regression → averaged word2vec + feedforward network →
fine-tuned BERT → distilled and quantized student (ONNX Runtime).

Setup and run instructions are added as the project is built.

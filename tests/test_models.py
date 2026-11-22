import numpy as np

from router.evaluation.metrics import apply_threshold, intent_metrics, tune_threshold
from router.models.tfidf import TfidfModel
from router.models.w2v_ffn import W2VFFNModel, average_vectors, train_ffn, train_word2vec

TEXTS = ["my dishwasher will not drain", "dishwasher is leaking water", "bulaşık makinem çalışmıyor",
         "what is my bank balance", "check my account balance", "hesap bakiyem ne kadar"] * 5
INTENTS = ["appliance_fault"] * 3 + ["balance"] * 3
INTENTS = INTENTS * 5


def test_tfidf_roundtrip(tmp_path):
    model = TfidfModel.train(TEXTS, INTENTS, C=10)
    assert model.predict(["dishwasher not draining"])[0][0] == "appliance_fault"
    model.threshold = 0.42
    model.save(tmp_path / "m.joblib")
    loaded = TfidfModel.load(tmp_path / "m.joblib")
    assert loaded.threshold == 0.42
    assert np.allclose(loaded.predict_proba(TEXTS[:2]), model.predict_proba(TEXTS[:2]))


def test_threshold_sends_low_confidence_to_oos():
    probs = np.array([[0.9, 0.1], [0.55, 0.45]])
    assert apply_threshold(probs, ["balance", "appliance_fault"], 0.6) == ["balance", "oos"]


def test_metrics_and_tuning():
    gold = ["balance", "oos", "appliance_fault", "oos"]
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])
    labels = ["balance", "appliance_fault"]
    m = intent_metrics(gold, apply_threshold(probs, labels, 0.0))
    assert m["in_scope_accuracy"] == 1.0 and m["oos_recall"] == 0.0
    t, best = tune_threshold(probs, labels, gold)
    assert 0.7 < t <= 0.8 and best["oos_recall"] == 1.0


def test_w2v_ffn_roundtrip(tmp_path):
    w2v = train_word2vec(TEXTS, dim=16, epochs=5)
    vectors = {w: w2v.wv[w] for w in w2v.wv.index_to_key}
    labels = ["appliance_fault", "balance"]
    y = np.array([labels.index(i) for i in INTENTS])
    x = average_vectors(TEXTS, vectors, 16)
    net = train_ffn(x, y, x, y, n_classes=2, hidden=8, epochs=5, log=lambda _: None)
    model = W2VFFNModel(vectors, net, labels, threshold=0.3)
    model.save(tmp_path / "m.pt")
    loaded = W2VFFNModel.load(tmp_path / "m.pt")
    assert loaded.threshold == 0.3
    assert np.allclose(loaded.predict_proba(TEXTS[:3]), model.predict_proba(TEXTS[:3]), atol=1e-6)
    assert average_vectors(["zzz qqq"], vectors, 16).sum() == 0

"""Part III: sentiment classification from frozen LM representations.

Labelled data
  train: the first CLS_TRAIN reviews of the LM *training* split (the LMs saw this text,
         but never the labels)
  test : the LM *test* split (never seen by any LM or classifier during training)

Representations (all LMs frozen), each turned into a fixed-length vector from the
"features" layer (last RNN/LSTM layer or final Transformer LayerNorm):
  mean : average of token states over non-padding input positions (<bos> w1 .. wn)
  last : state at the last non-padding position (has read the whole review)

Classifier: StandardScaler + logistic regression, C chosen by 5-fold CV on the train set.
Run part1_language_models.py and part2_embeddings.py first.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import config as C
from data import prepare_data, PAD_ID
from models import build_model
from lm_utils import set_seed, save_json, load_json

# name -> (architecture, embedding matrix file or None)
LMS = {
    "rnn": ("rnn", None),
    "lstm": ("lstm", None),
    "transformer": ("transformer", None),
    "lstm_w2v": ("lstm", "emb_w2v.npy"),
    "lstm_glove": ("lstm", "emb_glove.npy"),
}


def load_lm(name, V):
    arch, emb_file = LMS[name]
    if emb_file is None:
        model = build_model(arch, V)
    else:
        M = np.load(os.path.join(C.MODEL_DIR, emb_file))
        model = build_model(arch, V, emb_matrix=M, trainable_emb=False)
    model.load_weights(os.path.join(C.MODEL_DIR, f"{name}.weights.h5"))
    return model


def extract_features(model, X, batch_size=256):
    """Return (mean-pooled, last-state) features, both of shape (N, H)."""
    feat_model = keras.Model(model.input, model.get_layer("features").output)
    means, lasts = [], []
    for i in range(0, len(X), batch_size):
        x = X[i:i + batch_size]
        h = feat_model(x, training=False).numpy()          # (B, L, H)
        mask = (x != PAD_ID).astype(np.float32)[..., None]  # (B, L, 1)
        means.append((h * mask).sum(1) / mask.sum(1))
        last_idx = (x != PAD_ID).sum(1) - 1
        lasts.append(h[np.arange(len(x)), last_idx])
    return np.concatenate(means), np.concatenate(lasts)


def fit_eval(Xtr, ytr, Xte, yte, scale=True):
    steps = [StandardScaler()] if scale else []
    pipe = make_pipeline(*steps, LogisticRegression(max_iter=3000))
    gs = GridSearchCV(pipe, {"logisticregression__C": C.CLS_C_GRID}, cv=5, n_jobs=-1)
    gs.fit(Xtr, ytr)
    pred = gs.predict(Xte)
    return pred, {"accuracy": accuracy_score(yte, pred),
                  "macro_f1": f1_score(yte, pred, average="macro"),
                  "best_C": gs.best_params_["logisticregression__C"],
                  "cv_accuracy": gs.best_score_,
                  "confusion": confusion_matrix(yte, pred).tolist()}


def main():
    set_seed(C.SEED)
    data = prepare_data()
    V = len(data["vocab"])
    tr = {k: data["train"][k][: C.CLS_TRAIN] for k in ("X", "labels", "text")}
    te = {k: data["test"][k] for k in ("X", "labels", "text")}
    ytr, yte = tr["labels"], te["labels"]
    print(f"classifier train {len(ytr)} (pos {ytr.mean():.2%}), test {len(yte)} (pos {yte.mean():.2%})")

    rows, preds = [], {}

    # ---- reference: TF-IDF bag of words (not an LM representation)
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    pred, m = fit_eval(tfidf.fit_transform(tr["text"]), ytr, tfidf.transform(te["text"]), yte,
                       scale=False)
    rows.append({"representation": "tfidf_bow (reference)", "lm": None, "pooling": None, **m})
    preds["tfidf_bow"] = pred

    # ---- LM representations
    ppl = {r["model"]: r["test_ppl"]
           for r in load_json(os.path.join(C.OUT_DIR, "part1_results.json"))["results"]}
    for r in load_json(os.path.join(C.OUT_DIR, "part2_results.json"))["results"]:
        if "skip-gram" in r["model"]:
            ppl["lstm_w2v"] = r["test_ppl"]
        elif "GloVe" in r["model"]:
            ppl["lstm_glove"] = r["test_ppl"]

    for name in LMS:
        model = load_lm(name, V)
        f_tr = extract_features(model, tr["X"])
        f_te = extract_features(model, te["X"])
        for pooling, a, b in [("mean", f_tr[0], f_te[0]), ("last", f_tr[1], f_te[1])]:
            pred, m = fit_eval(a, ytr, b, yte)
            key = f"{name}_{pooling}"
            preds[key] = pred
            rows.append({"representation": key, "lm": name, "pooling": pooling,
                         "lm_test_ppl": ppl.get(name), "dim": a.shape[1], **m})
            print(f"{key:20s} acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")
        keras.backend.clear_session()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "part3_results.csv"), index=False)

    # ---- does LM ranking agree with downstream ranking?
    ranking = {}
    for pooling in ["mean", "last"]:
        sub = df[df["pooling"] == pooling]
        rho, p = spearmanr(sub["lm_test_ppl"], sub["accuracy"])  # expect negative if lower ppl helps
        ranking[pooling] = {
            "by_lm_ppl": sub.sort_values("lm_test_ppl")["lm"].tolist(),
            "by_accuracy": sub.sort_values("accuracy", ascending=False)["lm"].tolist(),
            "spearman_ppl_vs_acc": rho, "p_value": p,
        }

    # ---- representative errors: best LM representation vs. a contrasting one
    lm_rows = df[df["lm"].notna()].sort_values("accuracy", ascending=False)
    best, worst = lm_rows.iloc[0]["representation"], lm_rows.iloc[-1]["representation"]
    snip = lambda t: t.replace("<br />", " ")[:300]
    errors = {"best": best, "contrast": worst, "examples": {}}
    cases = {
        f"{best} wrong": np.where(preds[best] != yte)[0],
        f"{best} right, {worst} wrong": np.where((preds[best] == yte) & (preds[worst] != yte))[0],
        f"{best} wrong, {worst} right": np.where((preds[best] != yte) & (preds[worst] == yte))[0],
    }
    rng = np.random.default_rng(C.SEED)
    for label, idx in cases.items():
        pick = rng.choice(idx, size=min(5, len(idx)), replace=False) if len(idx) else []
        errors["examples"][label] = {
            "count": int(len(idx)),
            "samples": [{"label": int(yte[i]), "pred_best": int(preds[best][i]),
                         "pred_contrast": int(preds[worst][i]),
                         "n_tokens": len(data["test"]["tokens"][i]),
                         "text": snip(te["text"][i])} for i in pick],
        }
    # error rate by review length (truncation to MAX_LEN may hurt long reviews)
    lengths = np.array([len(t) for t in data["test"]["tokens"]])
    long_mask = lengths >= C.MAX_LEN
    errors["error_rate_truncated_reviews"] = float((preds[best][long_mask] != yte[long_mask]).mean())
    errors["error_rate_short_reviews"] = float((preds[best][~long_mask] != yte[~long_mask]).mean())

    save_json({"results": rows, "ranking": ranking, "errors": errors},
              os.path.join(C.OUT_DIR, "part3_results.json"))

    print("\n" + df[["representation", "lm_test_ppl", "accuracy", "macro_f1", "best_C"]]
          .to_string(index=False))
    for pooling, r in ranking.items():
        print(f"\n[{pooling}] LM ranking (ppl):      {r['by_lm_ppl']}")
        print(f"[{pooling}] downstream ranking (acc): {r['by_accuracy']}")
        print(f"[{pooling}] Spearman(ppl, acc) = {r['spearman_ppl_vs_acc']:.2f}")


if __name__ == "__main__":
    main()

"""Part III error sample (report Appendix C).

Rebuilds the best Part III classifier (LSTM + own Skip-gram, mean pooling, C=1.0 as
chosen by cross-validation in part3_downstream.py) and writes 20 randomly sampled
test errors to outputs/part3_error_sample.txt, showing both the first MAX_LEN tokens
the model saw and the end of the full review. The errors were then categorised by hand.

Run after part3_downstream.py:  python part3_error_sample.py
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from data import prepare_data
from part3_downstream import load_lm, extract_features

N_SAMPLES = 20
SAMPLE_SEED = 0


def main():
    data = prepare_data()
    V = len(data["vocab"])
    ytr = data["train"]["labels"][: C.CLS_TRAIN]
    yte = data["test"]["labels"]

    model = load_lm("lstm_w2v", V)
    f_tr = extract_features(model, data["train"]["X"][: C.CLS_TRAIN])[0]
    f_te = extract_features(model, data["test"]["X"])[0]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(f_tr, ytr)
    pred = clf.predict(f_te)
    prob = clf.predict_proba(f_te)[:, 1]
    err = np.where(pred != yte)[0]
    print(f"accuracy {np.mean(pred == yte):.4f}, errors {len(err)}")

    rng = np.random.default_rng(SAMPLE_SEED)
    pick = np.sort(rng.choice(err, size=min(N_SAMPLES, len(err)), replace=False))
    lab = {0: "neg", 1: "pos"}
    path = os.path.join(C.OUT_DIR, "part3_error_sample.txt")
    with open(path, "w", encoding="utf-8") as f:
        for k, i in enumerate(pick, 1):
            toks = data["test"]["tokens"][i]
            full = data["test"]["text"][i].replace("<br />", " ")
            f.write(f"#{k} test_idx={i} true={lab[yte[i]]} pred={lab[pred[i]]} p(pos)={prob[i]:.2f} "
                    f"truncated={len(toks) >= C.MAX_LEN}\n")
            f.write(f"SEEN (first {C.MAX_LEN} tokens): {' '.join(toks)}\n")
            f.write(f"END OF FULL REVIEW: ...{full[-350:]}\n\n")
    print("written", path)


if __name__ == "__main__":
    main()

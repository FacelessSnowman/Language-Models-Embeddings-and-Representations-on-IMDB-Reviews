"""Part II: embedding ablation on the LSTM language model.

Settings (everything else identical: data, EMB_DIM=100, architecture, training procedure):
  lstm         random embeddings, trained jointly   (reused from Part I)
  lstm_w2v     own Skip-gram embeddings on the LM training split, frozen
  lstm_glove   public GloVe 6B 100d embeddings, frozen

Run part1_language_models.py first.
"""
import os
import time

import numpy as np
import pandas as pd

import config as C
from data import prepare_data, ensure_glove, PAD_ID, BOS, EOS
from models import build_model, count_params
from lm_utils import (set_seed, save_json, load_json, train_lm, evaluate_lm,
                      generate_all, generate_neural, plot_curves)

ARCH = "lstm"
PROBE_WORDS = ["good", "terrible", "actor", "movie", "funny", "boring", "plot", "music"]


# ---------------------------------------------------------------- embedding matrices
def fill_missing(matrix, found, rng):
    """Missing rows ~ N(0, std of found vectors); <pad> row is zero. Rows stay frozen."""
    std = matrix[found].std()
    missing = ~found
    matrix[missing] = rng.normal(0.0, std, size=(missing.sum(), C.EMB_DIM))
    matrix[PAD_ID] = 0.0
    return matrix


def coverage_stats(vocab, found):
    """Type coverage over the vocabulary and token coverage over the training corpus."""
    words = vocab.itos[4:]  # exclude specials
    f = found[4:]
    tok_total = sum(vocab.counts[w] for w in words)
    tok_found = sum(vocab.counts[w] for w, ok in zip(words, f) if ok)
    return {"type_coverage": float(f.mean()), "token_coverage": tok_found / tok_total,
            "n_missing_types": int((~f).sum()),
            "example_missing": [w for w, ok in zip(words, f) if not ok][:25]}


def train_word2vec(data, vocab):
    """Skip-gram on the same truncated training reviews (OOV -> <unk>, with <bos>/<eos>)."""
    from gensim.models import Word2Vec

    sentences = [[BOS] + [vocab.itos[i] for i in ids] + [EOS] for ids in data["train"]["ids"]]
    t0 = time.perf_counter()
    w2v = Word2Vec(sentences, vector_size=C.EMB_DIM, window=C.W2V_WINDOW, min_count=1,
                   sg=C.W2V_SG, negative=C.W2V_NEG, epochs=C.W2V_EPOCHS,
                   seed=C.SEED, workers=1)  # workers=1 for reproducibility
    cost = time.perf_counter() - t0

    rng = np.random.default_rng(C.SEED)
    M = np.zeros((len(vocab), C.EMB_DIM), dtype=np.float32)
    found = np.zeros(len(vocab), dtype=bool)
    for i, w in enumerate(vocab.itos):
        if w in w2v.wv:
            M[i] = w2v.wv[w]
            found[i] = True
    found[PAD_ID] = False
    M = fill_missing(M, found, rng)
    return M, found, cost


def load_glove(vocab):
    rng = np.random.default_rng(C.SEED + 1)
    M = np.zeros((len(vocab), C.EMB_DIM), dtype=np.float32)
    found = np.zeros(len(vocab), dtype=bool)
    t0 = time.perf_counter()
    with open(ensure_glove(), encoding="utf-8") as f:
        for line in f:
            word, *vec = line.rstrip().split(" ")
            idx = vocab.stoi.get(word)
            if idx is not None and idx >= 4:  # specials (<unk>, <bos>, ...) not in GloVe
                M[idx] = np.asarray(vec, dtype=np.float32)
                found[idx] = True
    load_time = time.perf_counter() - t0
    M = fill_missing(M, found, rng)
    return M, found, load_time


def nearest_neighbours(M, vocab, words, k=6):
    norm = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-8)
    out = {}
    for w in words:
        if w not in vocab.stoi:
            continue
        sims = norm @ norm[vocab.stoi[w]]
        sims[[PAD_ID, vocab.stoi[w]]] = -np.inf
        out[w] = [vocab.itos[j] for j in np.argsort(-sims)[:k]]
    return out


# ---------------------------------------------------------------- main
def main():
    set_seed(C.SEED)
    data = prepare_data()
    vocab = data["vocab"]
    V = len(vocab)

    # ---- 1. own embeddings (cost reported separately from LM training)
    M_w2v, found_w2v, w2v_time = train_word2vec(data, vocab)
    print(f"Skip-gram training: {w2v_time:.1f}s")
    # ---- 2. public embeddings
    M_glove, found_glove, glove_time = load_glove(vocab)
    cov = coverage_stats(vocab, found_glove)
    print(f"GloVe coverage: {cov['type_coverage']:.2%} of types, "
          f"{cov['token_coverage']:.2%} of training tokens")
    np.save(os.path.join(C.MODEL_DIR, "emb_w2v.npy"), M_w2v)
    np.save(os.path.join(C.MODEL_DIR, "emb_glove.npy"), M_glove)

    # ---- 3. random-embedding LSTM from Part I (same dim/arch/data/procedure)
    p1 = load_json(os.path.join(C.OUT_DIR, "part1_results.json"))["results"]
    row_rand = next(r for r in p1 if r["model"] == ARCH)
    row_rand = {**row_rand, "model": "lstm (random, trainable)"}
    hist_rand = load_json(os.path.join(C.OUT_DIR, f"history_{ARCH}.json"))
    hist_rand["name"] = "random (trainable)"
    gens = {"random": load_json(os.path.join(C.OUT_DIR, "part1_generations.json"))[ARCH]}

    # random embeddings after LM training, for the neighbour comparison
    rand_model = build_model(ARCH, V)
    rand_model.load_weights(os.path.join(C.MODEL_DIR, f"{ARCH}.weights.h5"))
    M_rand = rand_model.get_layer("embedding").get_weights()[0]

    rows, hists = [row_rand], [hist_rand]
    for tag, M, label in [("w2v", M_w2v, "lstm (own skip-gram, frozen)"),
                          ("glove", M_glove, "lstm (GloVe, frozen)")]:
        name = f"{ARCH}_{tag}"
        set_seed(C.SEED)
        model = build_model(ARCH, V, emb_matrix=M, trainable_emb=False)
        hist = train_lm(model, data, name)
        # sanity check: the whole input-embedding table stayed fixed
        assert np.allclose(model.get_layer("embedding").get_weights()[0], M), "embedding changed!"
        ev = evaluate_lm(model, data["test"]["X"], data["test"]["Y"])
        g, s_per_tok = generate_all(generate_neural, model, vocab)
        gens[tag] = g
        hist["name"] = label.split("(")[1].rstrip(")")
        hists.append(hist)
        rows.append({"model": label, **ev, **count_params(model),
                     "train_time_s": hist["train_time_s"], "epochs_run": hist["epochs_run"],
                     "best_epoch": hist["best_epoch"], "gen_s_per_token": s_per_tok})
        print(f"[{name}] test CE {ev['test_ce']:.4f}  PPL {ev['test_ppl']:.1f}")

    # ---- stability summary: epoch-to-epoch val-loss changes and train/val gap
    for r, h in zip(rows, hists):
        vl = np.array(h["val_loss"])
        r["val_loss_increases"] = int((np.diff(vl) > 0).sum())
        r["final_train_val_gap"] = float(h["val_loss"][-1] - h["loss"][-1])
        r["mean_epoch_time_s"] = float(np.mean(h["epoch_times"])) if h.get("epoch_times") else None

    # ---- save
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "part2_results.csv"), index=False)
    save_json({"results": rows,
               "embedding_cost": {"skipgram_train_time_s": w2v_time,
                                  "glove_load_time_s": glove_time},
               "w2v_coverage": coverage_stats(vocab, found_w2v),
               "glove_coverage": cov,
               "missing_init": "N(0, std of found vectors), frozen; <pad> = 0"},
              os.path.join(C.OUT_DIR, "part2_results.json"))
    save_json(gens, os.path.join(C.OUT_DIR, "part2_generations.json"))
    with open(os.path.join(C.OUT_DIR, "part2_generations.txt"), "w", encoding="utf-8") as f:
        for p in C.PROMPTS:
            f.write(f"PROMPT: {p}\n")
            for m, g in gens.items():
                f.write(f"  [{m:6s}] {g[p]}\n")
            f.write("\n")
    nn = {name: nearest_neighbours(M, vocab, PROBE_WORDS)
          for name, M in [("random_trained", M_rand), ("skipgram", M_w2v), ("glove", M_glove)]}
    save_json(nn, os.path.join(C.OUT_DIR, "part2_neighbours.json"))
    plot_curves(hists, os.path.join(C.FIG_DIR, "part2_loss_curves.png"),
                "Part II: LSTM with different input embeddings")

    cols = ["model", "test_ce", "test_ppl", "trainable_params", "frozen_params",
            "train_time_s", "epochs_run", "best_epoch", "val_loss_increases"]
    print("\n" + df[cols].to_string(index=False))
    print(f"\nSkip-gram training cost (separate from LM training): {w2v_time:.1f}s")


if __name__ == "__main__":
    main()

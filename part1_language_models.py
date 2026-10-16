"""Part I: n-gram vs RNN vs LSTM vs Transformer language models on IMDB reviews.

Outputs (in outputs/):
  part1_results.csv / .json   test CE/perplexity, params, training & inference cost
  part1_generations.txt       matched-decoding continuations for every prompt
  figures/part1_loss_curves.png
  history_<model>.json, models/<model>.weights.h5
"""
import os
import pickle
import time

import numpy as np
import pandas as pd

import config as C
from data import prepare_data
from ngram import InterpolatedTrigramLM
from models import build_model, count_params
from lm_utils import (set_seed, hardware_info, save_json, train_lm, evaluate_lm,
                      generate_all, generate_neural, generate_ngram, plot_curves)


def run_ngram(data, V):
    lm = InterpolatedTrigramLM(V, add_k=C.NGRAM_ADD_K)
    t0 = time.perf_counter()
    lm.fit(data["train"]["ids"])
    best_val_ce, lambdas = lm.tune(data["val"]["ids"])
    train_time = time.perf_counter() - t0
    print(f"[ngram] lambdas (uni, bi, tri) = {tuple(round(l, 2) for l in lambdas)}, "
          f"val CE = {best_val_ce:.4f}")

    t0 = time.perf_counter()
    ce, n_tok = lm.cross_entropy(data["test"]["ids"])
    infer = time.perf_counter() - t0
    gens, s_per_tok = generate_all(generate_ngram, lm, data["vocab"])

    with open(os.path.join(C.MODEL_DIR, "ngram.pkl"), "wb") as f:
        pickle.dump({"lambdas": lm.lambdas}, f)  # counts are cheap to rebuild
    row = {"model": "trigram (interp.)", "test_ce": ce, "test_ppl": float(np.exp(ce)),
           "test_tokens": n_tok, "trainable_params": lm.num_entries(), "frozen_params": 0,
           "total_params": lm.num_entries(), "train_time_s": train_time, "epochs_run": None,
           "best_epoch": None, "infer_time_s": infer, "tokens_per_s": n_tok / infer,
           "gen_s_per_token": s_per_tok, "lambdas": lambdas}
    return row, gens


def run_neural(kind, data, V):
    set_seed(C.SEED)
    model = build_model(kind, V)
    model.summary()
    hist = train_lm(model, data, kind)
    ev = evaluate_lm(model, data["test"]["X"], data["test"]["Y"])
    gens, s_per_tok = generate_all(generate_neural, model, data["vocab"])
    row = {"model": kind, **ev, **count_params(model),
           "train_time_s": hist["train_time_s"], "epochs_run": hist["epochs_run"],
           "best_epoch": hist["best_epoch"], "gen_s_per_token": s_per_tok}
    print(f"[{kind}] test CE {ev['test_ce']:.4f}  PPL {ev['test_ppl']:.1f}")
    return row, gens, hist


def main():
    set_seed(C.SEED)
    hw = hardware_info()
    print("Hardware:", hw)
    data = prepare_data()
    V = len(data["vocab"])

    rows, generations, histories = [], {}, []

    row, gens = run_ngram(data, V)
    rows.append(row)
    generations["trigram"] = gens

    for kind in ["rnn", "lstm", "transformer"]:
        row, gens, hist = run_neural(kind, data, V)
        rows.append(row)
        generations[kind] = gens
        histories.append(hist)

    # ------------------------------------------------------------ save results
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "part1_results.csv"), index=False)
    save_json({"hardware": hw, "results": rows}, os.path.join(C.OUT_DIR, "part1_results.json"))
    save_json(generations, os.path.join(C.OUT_DIR, "part1_generations.json"))
    with open(os.path.join(C.OUT_DIR, "part1_generations.txt"), "w", encoding="utf-8") as f:
        f.write(f"Decoding: temperature={C.TEMPERATURE}, top_k={C.TOP_K}, "
                f"length={C.GEN_LEN} tokens, seed for prompt i = {C.SEED}+i, "
                "special tokens (<unk>, <eos>, ...) banned\n\n")
        for p in C.PROMPTS:
            f.write(f"PROMPT: {p}\n")
            for m, g in generations.items():
                f.write(f"  [{m:11s}] {g[p]}\n")
            f.write("\n")
    plot_curves(histories, os.path.join(C.FIG_DIR, "part1_loss_curves.png"),
                "Part I: training (dashed) vs validation (solid) loss")

    cols = ["model", "test_ce", "test_ppl", "total_params", "train_time_s",
            "epochs_run", "tokens_per_s", "gen_s_per_token"]
    print("\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

"""Data loading, tokenisation, vocabulary and LM tensors for IMDB reviews."""
import os
import re
import glob
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import config as C

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


# ---------------------------------------------------------------- download helpers
def ensure_imdb():
    """Return the path to 'IMDB Dataset.csv', downloading it with kagglehub if needed."""
    if os.path.exists(C.IMDB_CSV):
        return C.IMDB_CSV
    try:
        import kagglehub
        path = kagglehub.dataset_download(C.IMDB_KAGGLE)
        return os.path.join(path, "IMDB Dataset.csv")
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not find {C.IMDB_CSV} and kagglehub download failed ({e}). "
            f"Download it manually from https://www.kaggle.com/datasets/{C.IMDB_KAGGLE} "
            f"and place 'IMDB Dataset.csv' in '{C.DATA_DIR}/'."
        )


def ensure_glove():
    """Return the path to glove.6B.100d.txt, downloading it with kagglehub if needed."""
    if os.path.exists(C.GLOVE_PATH):
        return C.GLOVE_PATH
    try:
        import kagglehub
        path = kagglehub.dataset_download(C.GLOVE_KAGGLE)
        hits = glob.glob(os.path.join(path, "**", "glove.6B.100d.txt"), recursive=True)
        if hits:
            return hits[0]
        raise FileNotFoundError("glove.6B.100d.txt not in the Kaggle download")
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not find {C.GLOVE_PATH} ({e}). Download glove.6B.zip from "
            "https://nlp.stanford.edu/data/glove.6B.zip and extract glove.6B.100d.txt "
            f"into '{C.DATA_DIR}/'."
        )


# ---------------------------------------------------------------- tokenisation
_TAG = re.compile(r"<[^>]+>")                      # IMDB reviews contain <br /> tags
_TOKEN = re.compile(r"n't|'\w+|[^\W_]+|[^\w\s]")   # PTB-like: "don't" -> do n't, "it's" -> it 's


def tokenize(text):
    """Lower-case, strip HTML, split clitics the same way GloVe's tokenizer does."""
    text = _TAG.sub(" ", text.lower())
    text = text.replace("n't", " n't")
    return _TOKEN.findall(text)


# ---------------------------------------------------------------- vocabulary
class Vocab:
    def __init__(self, token_lists, size):
        counts = Counter(t for toks in token_lists for t in toks)
        most_common = [w for w, _ in counts.most_common(size - len(SPECIALS))]
        self.itos = SPECIALS + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = counts

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)


def to_lm_arrays(id_lists, max_len=C.MAX_LEN):
    """Build next-token (X, Y) pairs, right-padded with PAD_ID.

    seq = <bos> w1 .. wn <eos>;  X = seq[:-1], Y = seq[1:].
    Position t of Y is predicted from X[:t+1] only (causal).
    """
    L = max_len + 1
    X = np.full((len(id_lists), L), PAD_ID, dtype=np.int32)
    Y = np.full((len(id_lists), L), PAD_ID, dtype=np.int32)
    for i, ids in enumerate(id_lists):
        seq = [BOS_ID] + ids[:max_len] + [EOS_ID]
        X[i, : len(seq) - 1] = seq[:-1]
        Y[i, : len(seq) - 1] = seq[1:]
    return X, Y


# ---------------------------------------------------------------- main entry point
def prepare_data(use_cache=True):
    """Load IMDB, de-duplicate, split, tokenise, build vocab and LM arrays.

    Returns a dict with keys: vocab, train, val, test. Each split is a dict with
    text, tokens (truncated to MAX_LEN), ids, labels (0/1), X, Y.
    """
    if use_cache and os.path.exists(C.DATA_CACHE):
        with open(C.DATA_CACHE, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(ensure_imdb())
    n_raw = len(df)
    df = df.drop_duplicates(subset="review").reset_index(drop=True)  # avoid train/test leakage
    df = df.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    print(f"IMDB: {n_raw} reviews, {len(df)} after de-duplication")

    bounds = {
        "train": (0, C.N_TRAIN),
        "val": (C.N_TRAIN, C.N_TRAIN + C.N_VAL),
        "test": (C.N_TRAIN + C.N_VAL, C.N_TRAIN + C.N_VAL + C.N_TEST),
    }
    splits = {}
    for name, (a, b) in bounds.items():
        part = df.iloc[a:b]
        toks = [tokenize(t)[: C.MAX_LEN] for t in part["review"]]
        splits[name] = {
            "text": part["review"].tolist(),
            "tokens": toks,
            "labels": (part["sentiment"] == "positive").astype(int).to_numpy(),
        }

    vocab = Vocab(splits["train"]["tokens"], C.VOCAB_SIZE)  # train split only
    for name, s in splits.items():
        s["ids"] = [vocab.encode(t) for t in s["tokens"]]
        s["X"], s["Y"] = to_lm_arrays(s["ids"])
        n_tok = int((s["Y"] != PAD_ID).sum())
        n_unk = int((s["Y"] == UNK_ID).sum())
        print(f"{name:5s}: {len(s['ids'])} reviews, {n_tok} target tokens, "
              f"OOV rate {n_unk / n_tok:.2%}")

    data = {"vocab": vocab, **splits}
    with open(C.DATA_CACHE, "wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    d = prepare_data(use_cache=False)
    print("vocab size:", len(d["vocab"]))
    print("example tokens:", d["train"]["tokens"][0][:30])

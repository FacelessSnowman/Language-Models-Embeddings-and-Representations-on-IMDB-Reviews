"""Shared training, evaluation, timing and generation utilities."""
import json
import os
import platform
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config as C
from data import PAD_ID, UNK_ID, BOS_ID, EOS_ID, tokenize


# ---------------------------------------------------------------- misc
def set_seed(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)  # python, numpy and tensorflow


def hardware_info():
    gpus = tf.config.list_physical_devices("GPU")
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "tensorflow": tf.__version__, "gpus": [g.name for g in gpus]}
    try:
        if gpus:
            info["gpu_details"] = tf.config.experimental.get_device_details(gpus[0])
    except Exception:  # noqa: BLE001
        pass
    return info


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- loss (padding excluded)
def masked_ce(y_true, logits):
    """Mean next-token cross-entropy over non-padding targets only."""
    y_true = tf.cast(y_true, tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), loss.dtype)
    return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def make_dataset(X, Y, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=C.SEED, reshuffle_each_iteration=True)
    return ds.batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- training
class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t)


class KeepBestWeights(keras.callbacks.Callback):
    """Restore lowest-val-loss weights at the end, whether or not early stopping fired
    (tf.keras 2.x EarlyStopping only restores when it actually stops)."""

    def on_train_begin(self, logs=None):
        self.best, self.weights = np.inf, None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best:
            self.best, self.weights = logs["val_loss"], self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.weights is not None:
            self.model.set_weights(self.weights)


def train_lm(model, data, name):
    """Identical training procedure for every neural LM (Adam, early stopping on val loss).

    Returns a dict with the loss history, per-epoch times and total training time.
    Best weights (lowest val loss) are restored and saved to outputs/models/.
    """
    model.compile(optimizer=keras.optimizers.Adam(C.LR, clipnorm=C.CLIPNORM), loss=masked_ce)
    train_ds = make_dataset(data["train"]["X"], data["train"]["Y"], shuffle=True)
    val_ds = make_dataset(data["val"]["X"], data["val"]["Y"], shuffle=False)
    timer = EpochTimer()
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.PATIENCE)
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=C.MAX_EPOCHS,
                     callbacks=[timer, stopper, KeepBestWeights()], verbose=2)
    train_time = time.perf_counter() - t0

    model.save_weights(os.path.join(C.MODEL_DIR, f"{name}.weights.h5"))
    record = {
        "name": name,
        "loss": [float(v) for v in hist.history["loss"]],
        "val_loss": [float(v) for v in hist.history["val_loss"]],
        "epoch_times": timer.times,
        "train_time_s": train_time,
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
    }
    save_json(record, os.path.join(C.OUT_DIR, f"history_{name}.json"))
    return record


# ---------------------------------------------------------------- evaluation
def evaluate_lm(model, X, Y, batch_size=C.BATCH_SIZE):
    """Exact token-weighted test cross-entropy / perplexity, plus inference throughput.

    The common inference workload is: score every review in the test split with
    batch size BATCH_SIZE (one forward pass per batch). One warm-up batch is run
    first so graph tracing / cuDNN initialisation is not counted.
    """
    model(X[:batch_size], training=False)  # warm-up
    total_nll, total_tok = 0.0, 0
    t0 = time.perf_counter()
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size], training=False)
        y = Y[i:i + batch_size]
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits).numpy()
        mask = y != PAD_ID
        total_nll += float(nll[mask].sum())
        total_tok += int(mask.sum())
    elapsed = time.perf_counter() - t0
    ce = total_nll / total_tok
    return {"test_ce": ce, "test_ppl": float(np.exp(ce)), "test_tokens": total_tok,
            "infer_time_s": elapsed, "tokens_per_s": total_tok / elapsed}


# ---------------------------------------------------------------- generation
BANNED = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]  # EOS banned so every continuation has GEN_LEN tokens


def sample_next(probs_or_logits, rng, is_logits=True):
    """Temperature + top-k sampling with special tokens banned (same for every model)."""
    if is_logits:
        logp = probs_or_logits.astype(np.float64) / C.TEMPERATURE
    else:
        logp = np.log(np.maximum(probs_or_logits, 1e-30)) / C.TEMPERATURE
    logp[BANNED] = -np.inf
    top = np.argpartition(logp, -C.TOP_K)[-C.TOP_K:]
    p = np.exp(logp[top] - logp[top].max())
    p /= p.sum()
    return int(rng.choice(top, p=p))


def encode_prompt(vocab, prompt):
    return vocab.encode(tokenize(prompt))


def generate_neural(model, vocab, prompt, seed):
    rng = np.random.default_rng(seed)
    ids = [BOS_ID] + encode_prompt(vocab, prompt)
    n_prompt = len(ids)
    for _ in range(C.GEN_LEN):
        ctx = np.array([ids[-(C.MAX_LEN + 1):]], dtype=np.int32)  # Transformer max context
        logits = model(ctx, training=False)[0, -1].numpy()
        ids.append(sample_next(logits, rng, is_logits=True))
    return vocab.decode(ids[n_prompt:])


def generate_ngram(lm, vocab, prompt, seed):
    rng = np.random.default_rng(seed)
    ids = encode_prompt(vocab, prompt)
    n_prompt = len(ids)
    for _ in range(C.GEN_LEN):
        ids.append(sample_next(lm.next_dist(ids), rng, is_logits=False))
    return vocab.decode(ids[n_prompt:])


def generate_all(gen_fn, *args):
    """Continuations for every prompt; prompt i always uses seed SEED + i."""
    out = {}
    t0 = time.perf_counter()
    for i, p in enumerate(C.PROMPTS):
        out[p] = gen_fn(*args, p, C.SEED + i)
    elapsed = time.perf_counter() - t0
    return out, elapsed / (len(C.PROMPTS) * C.GEN_LEN)  # seconds per generated token


# ---------------------------------------------------------------- plotting
def plot_curves(histories, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, h in enumerate(histories):
        color = f"C{i}"
        ep = np.arange(1, len(h["loss"]) + 1)
        ax.plot(ep, h["loss"], "--", color=color, label=f"{h['name']} train")
        ax.plot(ep, h["val_loss"], "-o", color=color, ms=4, label=f"{h['name']} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy (nats / token)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

"""TensorFlow / Keras language models: SimpleRNN, LSTM and a causal Transformer.

Each model maps token ids (B, L) -> next-token logits (B, L, V). The layer named
"features" exposes per-token hidden states used in Part III.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config as C


def make_embedding(vocab_size, emb_matrix=None, trainable=True):
    """Random trainable embedding, or one initialised from `emb_matrix` (optionally frozen)."""
    if emb_matrix is None:
        return layers.Embedding(vocab_size, C.EMB_DIM, name="embedding")
    assert emb_matrix.shape == (vocab_size, C.EMB_DIM)
    return layers.Embedding(
        vocab_size, C.EMB_DIM,
        embeddings_initializer=keras.initializers.Constant(emb_matrix),
        trainable=trainable, name="embedding",
    )


def build_recurrent_lm(vocab_size, cell="lstm", emb_matrix=None, trainable_emb=True):
    inp = keras.Input(shape=(None,), dtype="int32", name="tokens")
    x = make_embedding(vocab_size, emb_matrix, trainable_emb)(inp)
    x = layers.Dropout(C.DROPOUT)(x)
    rnn = layers.LSTM if cell == "lstm" else layers.SimpleRNN
    h = rnn(C.HIDDEN, return_sequences=True, name="features")(x)
    h = layers.Dropout(C.DROPOUT)(h)
    logits = layers.Dense(vocab_size, name="logits")(h)
    return keras.Model(inp, logits, name=f"{cell}_lm")


class PositionalEmbedding(layers.Layer):
    """Learned absolute position embeddings added to token embeddings."""

    def __init__(self, max_len, dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len, self.dim = max_len, dim
        self.pos = layers.Embedding(max_len, dim)

    def call(self, x):
        positions = tf.range(tf.shape(x)[1])
        return x + self.pos(positions)

    def get_config(self):
        return {**super().get_config(), "max_len": self.max_len, "dim": self.dim}


def transformer_block(x, idx):
    """Pre-LayerNorm decoder block with causal multi-head self-attention."""
    h = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{idx}")(x)
    h = layers.MultiHeadAttention(
        num_heads=C.TF_HEADS, key_dim=C.EMB_DIM // C.TF_HEADS,
        dropout=C.DROPOUT, name=f"mha_{idx}",
    )(h, h, use_causal_mask=True)  # position t attends only to positions <= t
    h = layers.Dropout(C.DROPOUT)(h)
    x = layers.Add()([x, h])
    h = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{idx}")(x)
    h = layers.Dense(C.TF_FF, activation="gelu")(h)
    h = layers.Dense(C.EMB_DIM)(h)
    h = layers.Dropout(C.DROPOUT)(h)
    return layers.Add()([x, h])


def build_transformer_lm(vocab_size, emb_matrix=None, trainable_emb=True):
    inp = keras.Input(shape=(None,), dtype="int32", name="tokens")
    x = make_embedding(vocab_size, emb_matrix, trainable_emb)(inp)
    x = PositionalEmbedding(C.MAX_LEN + 1, C.EMB_DIM, name="pos_embedding")(x)
    x = layers.Dropout(C.DROPOUT)(x)
    for i in range(C.TF_LAYERS):
        x = transformer_block(x, i)
    h = layers.LayerNormalization(epsilon=1e-6, name="features")(x)
    logits = layers.Dense(vocab_size, name="logits")(h)
    return keras.Model(inp, logits, name="transformer_lm")


def build_model(kind, vocab_size, emb_matrix=None, trainable_emb=True):
    """kind in {'rnn', 'lstm', 'transformer'}."""
    if kind == "transformer":
        return build_transformer_lm(vocab_size, emb_matrix, trainable_emb)
    return build_recurrent_lm(vocab_size, kind, emb_matrix, trainable_emb)


def count_params(model):
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    frozen = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {"trainable_params": trainable, "frozen_params": frozen,
            "total_params": trainable + frozen}

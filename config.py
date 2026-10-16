"""Central configuration for all experiments (Parts I-III).

Every script imports from here so that data, model sizes and training
procedure stay identical across the comparisons.
"""
import os

# ---------------------------------------------------------------- reproducibility
SEED = 42

# QUICK=1 shrinks data/epochs for a fast end-to-end smoke test (results not meaningful)
QUICK = os.environ.get("QUICK", "0") == "1"

# ---------------------------------------------------------------- paths
DATA_DIR = "data"
OUT_DIR = "outputs_quick" if QUICK else "outputs"
MODEL_DIR = os.path.join(OUT_DIR, "models")
FIG_DIR = os.path.join(OUT_DIR, "figures")
for _d in (DATA_DIR, OUT_DIR, MODEL_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
IMDB_KAGGLE = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
IMDB_CSV = os.path.join(DATA_DIR, "IMDB Dataset.csv")

# Kaggle: https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt
# (original source: https://nlp.stanford.edu/data/glove.6B.zip)
GLOVE_KAGGLE = "danielwillgeorge/glove6b100dtxt"
GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B.100d.txt")

DATA_CACHE = os.path.join(OUT_DIR, "data_cache.pkl")

# ---------------------------------------------------------------- data
# Reviews are shuffled once (after de-duplication) and split into disjoint sets.
N_TRAIN = 20000      # LM training reviews
N_VAL = 2500         # LM validation reviews (early stopping, n-gram lambda tuning)
N_TEST = 2500        # LM test reviews (also the Part III test set)
MAX_LEN = 128        # keep the first MAX_LEN word tokens of each review
VOCAB_SIZE = 10000   # including the 4 special tokens

# ---------------------------------------------------------------- models
EMB_DIM = 100        # matches GloVe 6B 100d so Part II can reuse the Part I LSTM
HIDDEN = 256         # RNN / LSTM hidden size
TF_LAYERS = 2        # Transformer blocks
TF_HEADS = 4         # attention heads (EMB_DIM must be divisible by this)
TF_FF = 256          # Transformer feed-forward width
DROPOUT = 0.1

NGRAM_ORDER = 3      # interpolated trigram
NGRAM_ADD_K = 1.0    # add-k smoothing for the unigram component

# ---------------------------------------------------------------- training (shared by every neural LM)
BATCH_SIZE = 64
LR = 1e-3
CLIPNORM = 1.0
MAX_EPOCHS = 10
PATIENCE = 2         # early stopping on val loss, best weights restored

# ---------------------------------------------------------------- generation (matched across models)
PROMPTS = [
    "this movie was",
    "the acting in this film",
    "i would not recommend",
    "the story is about",
    "my favorite part was",
]
GEN_LEN = 30
TEMPERATURE = 0.8
TOP_K = 40

# ---------------------------------------------------------------- Part II: own embeddings (Skip-gram)
W2V_SG = 1           # 1 = Skip-gram, 0 = CBOW
W2V_WINDOW = 5
W2V_EPOCHS = 5
W2V_NEG = 5

# ---------------------------------------------------------------- Part III
CLS_TRAIN = 5000     # labelled reviews drawn from the LM *train* split
CLS_C_GRID = [0.01, 0.1, 1.0, 10.0]

if QUICK:
    N_TRAIN, N_VAL, N_TEST = 2000, 300, 300
    VOCAB_SIZE = 3000
    MAX_EPOCHS = 2
    W2V_EPOCHS = 1
    CLS_TRAIN = 1000

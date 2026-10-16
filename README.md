# DSA4213 Assignment 1: Language Models, Embeddings, and Representations

**Kho Zhe Ming (A0271660H).** The report is in [`report/`](report/).

This project compares four language models on IMDB movie reviews: an interpolated trigram model, a SimpleRNN, an LSTM and a causal Transformer (**Part I**). It then tests three input-embedding settings for the LSTM: random and trainable, our own frozen Skip-gram vectors, and frozen GloVe vectors (**Part II**). Finally, it uses the frozen LM representations for sentiment classification (**Part III**).

The research question is whether **better next-token prediction leads to better review representations for sentiment classification**.

All neural models are written in TensorFlow/Keras.

## Main results (full run, seed 42, laptop CPU)

| Model | Test PPL | Train time | Sentiment accuracy (mean / last pooling) |
|---|---|---|---|
| Trigram (interpolated) | 133.4 | 9.6 s | – |
| RNN | 108.2 | 47.4 min | 69.4 / 53.6 |
| LSTM (random embeddings) | 122.1 | 58.5 min | 69.8 / 54.6 |
| Transformer | **91.1** | 34.3 min | 75.2 / **73.8** |
| LSTM + own Skip-gram (frozen) | 122.6 | 54.9 min | **79.1** / 60.6 |
| LSTM + GloVe 6B-100d (frozen) | 132.2 | 54.6 min | 76.2 / 56.2 |
| TF-IDF + logistic regression (reference) | – | – | 88.9 |

Lower perplexity did **not** produce better sentiment features. Pooling and the choice of input embedding mattered more.

## Data

| Resource | Source | Used for |
|---|---|---|
| IMDB 50K Movie Reviews | Kaggle: [`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (orig. Maas et al., 2011) | LM corpus (Parts I–II), sentiment labels (Part III) |
| GloVe 6B 100d | Kaggle: [`danielwillgeorge/glove6b100dtxt`](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt) (orig. [Stanford NLP](https://nlp.stanford.edu/projects/glove/), Pennington et al., 2014) | Public frozen embeddings (Part II) |

Both are downloaded automatically with `kagglehub` on first use; public datasets don't need a Kaggle account. For a manual setup, place `IMDB Dataset.csv` and `glove.6B.100d.txt` in `data/` instead. **The data files are not included in this repository** (see `.gitignore`).

### Preprocessing (`data.py`)
1. Drop duplicate reviews: 418 removed, leaving 49,582. This prevents train/test leakage. Then shuffle with seed 42.
2. Split into disjoint sets: **20,000 train / 2,500 validation / 2,500 test** reviews. The remaining 24,582 are unused.
3. Lower-case the text, strip HTML tags such as `<br />`, and tokenize like PTB/GloVe: `don't` becomes `do n't` and `it's` becomes `it 's`. Punctuation marks are separate tokens.
4. Keep the **first 128 tokens** of each review. The median review has 200 tokens, so 85% of reviews are truncated.
5. Build the vocabulary from the **training split only**: the 9,996 most frequent words plus `<pad> <unk> <bos> <eos>`. The OOV rate is 4.2% on train and 4.5% on test.
6. Turn each review into `<bos> w1 … wn <eos>`, with input `X = seq[:-1]` and target `Y = seq[1:]`. Sequences are right-padded, and **padding targets are excluded** from the loss and from every metric.

## Common evaluation setup
- Each target token is predicted only from preceding tokens **within the same review**. The trigram model uses the last two of them. The neural models use the whole review prefix (at most 128 tokens), through recurrence or the causal attention mask.
- Every model is scored on the **same 307,996 test targets**, including `<eos>`. The metric is token-weighted average cross-entropy in nats, and perplexity = exp(CE).
- **Inference workload:** score the full test split at batch size 64, after one warm-up batch, timed with wall-clock `time.perf_counter`. Generation speed is reported in seconds per generated token; it recomputes the whole prefix at every step.
- **Decoding** is the same for every model: temperature 0.8, top-k 40, 30 new tokens, and seed 42 + i for prompt i. `<unk>`, `<eos>`, `<pad>` and `<bos>` are banned so that every continuation has the same length.

## Models (`models.py`, `ngram.py`, `config.py`)
| Model | Details |
|---|---|
| Trigram | Interpolated trigram, bigram and unigram (add-1). λ is grid-searched on the validation set (chosen: 0.20 / 0.45 / 0.35). A component whose context is unseen is dropped and the remaining weights are renormalized. |
| RNN | Embedding(100) → SimpleRNN(256) → Dense(V), dropout 0.1 |
| LSTM | Embedding(100) → LSTM(256) → Dense(V), dropout 0.1 |
| Transformer | Embedding(100) + learned positions → 2 pre-LN blocks (4 heads, FFN 256, causal mask) → LN → Dense(V) |

**Shared training procedure:** Adam (lr 1e-3, clipnorm 1.0), batch size 64, at most 10 epochs, and early stopping on validation loss (patience 2) with the best weights restored. Part II reuses this procedure unchanged. In the full run no model stopped early; all were still improving at epoch 10.

**Part II embeddings:** dimension 100 throughout, chosen to match GloVe 100d.
- Our own embeddings are Skip-gram vectors from gensim Word2Vec (window 5, 5 negative samples, 5 epochs, seed 42, one worker). They are trained on the same truncated training reviews, with out-of-vocabulary words mapped to `<unk>`.
- Words with no pretrained vector, including `<unk>`, `<bos>` and `<eos>` in GloVe, are initialized from N(0, σ), where σ is the standard deviation of the vectors that were found. These rows are also frozen. `<pad>` is set to 0.
- A check confirms that the embedding table is unchanged after training.

**Part III:** the classifier is trained on 5,000 labelled reviews from the LM training split and tested on the 2,500 LM test reviews. Features come from the frozen `features` layer and are pooled in one of two ways: a mean over non-pad positions, or the state at the last non-pad position. The classifier is StandardScaler followed by LogisticRegression, with C ∈ {0.01, 0.1, 1, 10} chosen by 5-fold CV. A TF-IDF + logistic regression model, which sees the full review, is included as a reference. `part3_error_sample.py` rebuilds the best classifier and samples 20 of its errors (seed 0), which were then categorised by hand for Appendix C of the report.

## Setup and commands
The pinned dependencies in `requirements.txt` require **Python 3.12**. TensorFlow 2.18 doesn't support Python 3.13 or newer. Both options below were tested end to end on Windows 11.

### Option A: Docker
Requires Docker Desktop to be running. The image is built on `python:3.12-slim`. The project folder is mounted into the container, so `data/`, `outputs/` and the Kaggle download cache stay on the host, and code edits don't need a rebuild. The container runs on CPU.

**PowerShell (Windows):**
```powershell
docker compose build                                       # once, or after changing requirements.txt
$env:QUICK = "1"; docker compose run --rm dsa4213          # smoke test, a few minutes -> outputs_quick/
Remove-Item Env:QUICK
docker compose run --rm dsa4213                            # full run: all parts -> outputs/
docker compose run --rm dsa4213 python run_all.py 3 4      # rerun selected steps only
```

**bash (Linux/macOS/Git Bash):**
```bash
docker compose build
QUICK=1 docker compose run --rm dsa4213
docker compose run --rm dsa4213
```

If kagglehub asks for credentials, set `KAGGLE_USERNAME` and `KAGGLE_KEY` before running; they are passed through by compose.

### Option B: local virtual environment
**PowerShell (Windows):**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:KAGGLEHUB_CACHE = "data\kagglehub"   # optional: keep downloads in data/ (shared with Docker)
$env:QUICK = "1"; python run_all.py      # smoke test -> outputs_quick/
Remove-Item Env:QUICK                    # important: otherwise the "full" run stays in quick mode
python run_all.py                        # full run -> outputs/
```
If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe run_all.py` directly.

**bash (Linux/macOS):**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
QUICK=1 python run_all.py
python run_all.py
```

### Pipeline steps
`run_all.py` runs these in order. Pass step numbers to run a subset, for example `python run_all.py 2 3`.

| Step | Script | Produces | Full-run time* |
|---|---|---|---|
| 1 | `part1_language_models.py` | `outputs/part1_*`, `figures/part1_loss_curves.png` | ~2 h 20 min |
| 2 | `part2_embeddings.py` | `outputs/part2_*`, `figures/part2_loss_curves.png` | ~1 h 50 min |
| 3 | `part3_downstream.py` | `outputs/part3_results.*` | ~3 min |
| 4 | `part3_error_sample.py` | `outputs/part3_error_sample.txt` | ~2 min |

\*Measured on an Intel Core 7 laptop CPU with 16 GB RAM and no GPU.

The steps depend on each other: Part II reuses the Part I LSTM, and steps 3–4 load the trained LMs. `python data.py` prints split statistics on its own. With `QUICK=1`, everything is scaled down: 2k reviews, 2 epochs and a 3k vocabulary. Its output goes to `outputs_quick/` and is only meant to check that the pipeline runs. To change dataset size, model size or training budget, edit `config.py`, and delete the matching `data_cache.pkl` if you change preprocessing.

## Reproducibility
- The seed is 42 everywhere: Python, NumPy, TensorFlow (`keras.utils.set_random_seed`), the data shuffle, gensim and generation. The Part III error sample uses seed 0.
- CPU floating-point differences can still cause tiny run-to-run changes. For example, rebuilding the best classifier in step 4 gives 79.04% accuracy (524 errors) versus 79.08% (523) in step 3. Calling `tf.config.experimental.enable_op_determinism()` makes TensorFlow runs exact, but slower.
- Hardware and the TensorFlow version are recorded in `outputs/part1_results.json`.

## Outputs
| File | Content |
|---|---|
| `part1_results.csv` | Test CE and perplexity, parameter counts, training time, epochs, inference throughput and generation speed |
| `part1_generations.txt` | Continuations for each prompt, side by side |
| `history_<model>.json` | Training and validation loss and time per epoch |
| `figures/part{1,2}_loss_curves.png` | Training and validation loss curves |
| `part2_results.json` | Results for the embedding variants, GloVe and Skip-gram coverage, Skip-gram training cost, and stability statistics |
| `part2_neighbours.json` | Nearest neighbours of probe words for each embedding setting |
| `part3_results.json` | Accuracy and macro-F1 for each representation, LM perplexity ranking vs. downstream ranking (Spearman), and sampled errors |
| `part3_error_sample.txt` | The 20 errors categorised in the report's Appendix C |
| `models/*.weights.h5`, `models/emb_*.npy` | Trained LM weights and the frozen embedding matrices |

## Attribution
- Datasets and embeddings are listed above.
- Libraries: TensorFlow/Keras, gensim, scikit-learn, NumPy, pandas, SciPy and matplotlib.
- The code was drafted with the help of an AI coding assistant (Claude, Anthropic), then reviewed, run and verified by the author. See Appendix D of the report.

"""Interpolated trigram language model (Jelinek-Mercer style).

P(w | t, u) = [l3 * P_ML(w | t, u) + l2 * P_ML(w | u) + l1 * P_addk(w)] / Z

where a component is dropped (and Z renormalises) when its context was never
seen in training, so every conditional distribution sums to one. The lambdas
are tuned on the validation set by grid search.
"""
import itertools
from collections import Counter, defaultdict

import numpy as np

from data import PAD_ID, BOS_ID, EOS_ID


class InterpolatedTrigramLM:
    def __init__(self, vocab_size, add_k=1.0):
        self.V = vocab_size
        self.add_k = add_k
        self.lambdas = (0.1, 0.3, 0.6)  # (l1, l2, l3); overwritten by tune()
        self.banned = [PAD_ID, BOS_ID]  # never valid targets

    # ------------------------------------------------------------ training
    def fit(self, id_lists):
        """id_lists: one list of token ids per review, already truncated, no specials."""
        uni = np.zeros(self.V, dtype=np.float64)
        self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter)
        for ids in id_lists:
            h = [BOS_ID, BOS_ID] + list(ids) + [EOS_ID]
            for i in range(2, len(h)):
                t, u, w = h[i - 2], h[i - 1], h[i]
                uni[w] += 1
                self.bi[u][w] += 1
                self.tri[(t, u)][w] += 1
        self.bi_tot = {u: sum(c.values()) for u, c in self.bi.items()}
        self.tri_tot = {k: sum(c.values()) for k, c in self.tri.items()}

        p = uni + self.add_k
        p[self.banned] = 0.0
        self.p_uni = p / p.sum()
        return self

    def num_entries(self):
        """Stored parameters: unigram table + distinct bigram and trigram entries."""
        return (self.V + sum(len(c) for c in self.bi.values())
                + sum(len(c) for c in self.tri.values()))

    # ------------------------------------------------------------ scoring
    def component_probs(self, id_lists):
        """Per target token: (p1, p2, p3) with NaN where the context is unseen."""
        p1, p2, p3 = [], [], []
        for ids in id_lists:
            h = [BOS_ID, BOS_ID] + list(ids) + [EOS_ID]
            for i in range(2, len(h)):
                t, u, w = h[i - 2], h[i - 1], h[i]
                p1.append(self.p_uni[w])
                p2.append(self.bi[u][w] / self.bi_tot[u] if u in self.bi_tot else np.nan)
                p3.append(self.tri[(t, u)][w] / self.tri_tot[(t, u)]
                          if (t, u) in self.tri_tot else np.nan)
        return np.array(p1), np.array(p2), np.array(p3)

    @staticmethod
    def combine(p1, p2, p3, lambdas):
        l1, l2, l3 = lambdas
        num = l1 * p1
        den = np.full_like(p1, l1)
        m2, m3 = ~np.isnan(p2), ~np.isnan(p3)
        num[m2] += l2 * p2[m2]
        den[m2] += l2
        num[m3] += l3 * p3[m3]
        den[m3] += l3
        return num / den

    def tune(self, val_id_lists, step=0.05):
        """Grid-search (l1, l2, l3) on the simplex to minimise validation cross-entropy."""
        p1, p2, p3 = self.component_probs(val_id_lists)
        grid = np.arange(step, 1.0, step)
        best = (np.inf, None)
        for l1, l2 in itertools.product(grid, grid):
            l3 = 1.0 - l1 - l2
            if l3 < -1e-9:
                continue
            ce = -np.mean(np.log(self.combine(p1, p2, p3, (l1, l2, max(l3, 0.0)))))
            if ce < best[0]:
                best = (ce, (l1, l2, max(l3, 0.0)))
        self.lambdas = best[1]
        return best

    def cross_entropy(self, id_lists):
        """Average next-token cross-entropy (nats) over the same targets as the neural LMs."""
        p = self.combine(*self.component_probs(id_lists), self.lambdas)
        return float(-np.mean(np.log(p))), len(p)

    # ------------------------------------------------------------ generation
    def next_dist(self, context):
        """Full next-token distribution given the token-id history."""
        h = [BOS_ID, BOS_ID] + list(context)
        t, u = h[-2], h[-1]
        l1, l2, l3 = self.lambdas
        vec = l1 * self.p_uni
        den = l1
        if u in self.bi_tot:
            c = self.bi[u]
            v = np.zeros(self.V)
            v[list(c.keys())] = np.fromiter(c.values(), float) / self.bi_tot[u]
            vec = vec + l2 * v
            den += l2
        if (t, u) in self.tri_tot:
            c = self.tri[(t, u)]
            v = np.zeros(self.V)
            v[list(c.keys())] = np.fromiter(c.values(), float) / self.tri_tot[(t, u)]
            vec = vec + l3 * v
            den += l3
        return vec / den

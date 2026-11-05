"""Naive Bayes parameter search using training-partition validation."""
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src import pipeline as P
from src.data_io import load_bundled
from src.preprocess import make_ngrams

ROOT = Path(__file__).resolve().parent


def count_matrix(tokens):
    vocab = sorted({t for doc in tokens for t in doc})
    ids = {t: j for j, t in enumerate(vocab)}
    rows, cols, values = [], [], []
    for i, doc in enumerate(tokens):
        for term, count in Counter(doc).items():
            rows.append(i)
            cols.append(ids[term])
            values.append(count)
    return csr_matrix((values, (rows, cols)), shape=(len(tokens), len(vocab)), dtype=float)


def feature_rank(class_df, sizes):
    n = sizes.sum()
    a = class_df
    b = class_df.sum(axis=0)[None, :] - a
    c = sizes[:, None] - a
    d = n - a - b - c
    den = (a + b) * (c + d) * (a + c) * (b + d)
    chi = np.divide(n * (a * d - b * c) ** 2, den, out=np.zeros_like(a), where=den > 0)
    return chi.max(axis=0)


def main():
    docs, meta = load_bundled()
    labels = np.array(meta.category)
    classes = np.unique(labels)
    feats = P.build_features(docs.text.tolist(), P.PRESETS["D. B + lemmatisation (default)"])
    _, keep, _ = P.dedup_indices(docs, meta, feats["S"])
    a, _ = P.stratified_split(labels[keep], .2, 42)
    outer = np.array(keep)[a]
    records = []
    for ngram in (1, 2):
        X = count_matrix([make_ngrams(t, ngram) for t in feats["tokens"]])
        B = X.copy()
        B.data[:] = 1.
        for seed in (0, 1, 2):
            a, b = P.stratified_split(labels[outer], .2, seed)
            tr, va = outer[a], outer[b]
            sizes = np.array([(labels[tr] == c).sum() for c in classes])
            counts = np.stack([np.asarray(X[tr[labels[tr] == c]].sum(axis=0)).ravel() for c in classes])
            dfs = np.stack([np.asarray(B[tr[labels[tr] == c]].sum(axis=0)).ravel() for c in classes])
            rank = feature_rank(dfs, sizes)
            df = dfs.sum(axis=0)
            for min_df in (1, 3):
                allowed = np.where((df >= min_df) & (df < .95 * len(tr)))[0]
                ordered = allowed[np.argsort(-rank[allowed], kind="stable")]
                for top_k in (0, 3000, 10000, 30000):
                    selected = ordered[:top_k] if top_k else allowed
                    C, D = counts[:, selected], dfs[:, selected]
                    Xt, Bt = X[va][:, selected], B[va][:, selected]
                    for alpha in (.1, 1., 5.):
                        theta = (C + alpha) / (C.sum(axis=1, keepdims=True) + alpha * len(selected))
                        outside = C.sum(axis=0)[None, :] - C
                        complement = (outside + alpha) / (outside.sum(axis=1, keepdims=True) + alpha * len(selected))
                        presence = (D + alpha) / (sizes[:, None] + 2 * alpha)
                        scores = {
                            "multinomial": Xt @ np.log(theta).T + np.log(sizes / sizes.sum()),
                            "complement": Xt @ -np.log(complement).T,
                            "one_vs_rest": Xt @ (np.log(theta) - np.log(complement)).T + np.log(sizes / (sizes.sum() - sizes)),
                            "bernoulli": Bt @ (np.log(presence) - np.log1p(-presence)).T
                                         + np.log1p(-presence).sum(axis=1) + np.log(sizes / sizes.sum()),
                        }
                        for variant, score in scores.items():
                            pred = classes[np.asarray(score).argmax(axis=1)].tolist()
                            report = P.M.classification_report(labels[va].tolist(), pred, classes.tolist())
                            records.append(dict(ngram=ngram, min_df=min_df, top_k=top_k, alpha=alpha,
                                                variant=variant, seed=seed, accuracy=report["accuracy"],
                                                macro_f1=report["macro"]["f1"]))
            print("Finished training-only search: ngram", ngram, "seed", seed, flush=True)
        table = pd.DataFrame(records)
    rank = table.groupby(["ngram", "min_df", "top_k", "alpha", "variant"])[["accuracy", "macro_f1"]].mean()
    print(rank.sort_values("macro_f1", ascending=False).head(15).to_string(), flush=True)


if __name__ == "__main__":
    main()

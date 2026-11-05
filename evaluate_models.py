"""Evaluate classifier and clustering baselines: python evaluate_models.py."""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from src import pipeline as P
from src.data_io import load_bundled

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def main():
    docs, meta = load_bundled()
    labels = meta["category"].tolist()
    feats = P.build_features(docs["text"].tolist(), P.PRESETS["D. B + lemmatisation (default)"])
    _, keep, _ = P.dedup_indices(docs, meta, feats["S"])
    kept_labels = [labels[i] for i in keep]
    train, test = P.stratified_split(kept_labels, .2, 42)
    train, test = [keep[i] for i in train], [keep[i] for i in test]

    def nb(tr, te, variant, features="counts", alpha=1.):
        return P.run_naive_bayes(feats["tokens"], feats["index"], labels, tr, te, features, alpha, variant)

    # Every candidate is evaluated inside the outer training partition.
    validation = []
    for variant in ("multinomial", "complement"):
        for features in ("counts", "tfidf"):
            for alpha in (.1, .5, 1., 2.):
                for seed in (0, 1, 2):
                    a, b = P.stratified_split([labels[i] for i in train], .2, seed)
                    r = nb([train[i] for i in a], [train[i] for i in b], variant, features, alpha)["report"]
                    validation.append(dict(variant=variant, features=features, alpha=alpha, seed=seed,
                                           accuracy=r["accuracy"], macro_f1=r["macro"]["f1"]))
    val = pd.DataFrame(validation)
    val.to_csv(RES / "nb_validation.csv", index=False)
    ranked = val.groupby(["variant", "features", "alpha"])[["accuracy", "macro_f1"]].mean()
    print("Training-only validation leaders:\n", ranked.sort_values("macro_f1", ascending=False).head(4), flush=True)

    old = nb(train, test, "multinomial")
    new = nb(train, test, "complement")
    saved = pd.read_csv(RES / "baseline" / "nb_predictions.csv")
    assert saved["doc_id"].astype(str).tolist() == meta["doc_id"].iloc[test].astype(str).tolist()
    assert saved["predicted"].tolist() == old["y_pred"], "Baseline predictions no longer reproduce"
    a = np.array(old["y_pred"]) == np.array(old["y_true"])
    b = np.array(new["y_pred"]) == np.array(new["y_true"])
    gains, losses = int((~a & b).sum()), int((a & ~b).sum())
    discordant = gains + losses
    pvalue = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1)) / 2 ** discordant)
    pd.DataFrame([dict(corrected=gains, regressed=losses, exact_mcnemar_p=pvalue, test_documents=len(test))]).to_csv(
        RES / "nb_paired_test.csv", index=False)

    splits = []
    for seed in (0, 1, 2, 3, 4, 42):
        a, b = P.stratified_split(kept_labels, .2, seed)
        for variant in ("multinomial", "complement"):
            r = nb([keep[i] for i in a], [keep[i] for i in b], variant)["report"]
            splits.append(dict(seed=seed, variant=variant, accuracy=r["accuracy"], macro_f1=r["macro"]["f1"]))
    pd.DataFrame(splits).to_csv(RES / "nb_split_stability.csv", index=False)

    clustering = []
    for spherical in (False, True):
        for init in ("k-means++", "greedy k-means++"):
            for seed in (42, 7, 123):
                r = P.run_kmeans(feats["X"][keep], kept_labels, 5, init=init, seed=seed,
                                  n_init=5, spherical=spherical)
                assert np.isclose(r["sse"], min(r["all_runs_sse"]))
                clustering.append(dict(spherical=spherical, init=init, seed=seed, n_init=5,
                                       purity=r["purity"], rand_index=r["rand_index"], ari=r["ari"], sse=r["sse"]))
    km = pd.DataFrame(clustering)
    km.to_csv(RES / "kmeans_stability.csv", index=False)
    baseline_km = km[(~km.spherical) & (km.init == "k-means++") & (km.seed == 42)].iloc[0]
    improved_km = km[km.spherical & (km.init == "greedy k-means++") & (km.seed == 42)].iloc[0]
    rows = []
    for metric, before, after in (
        ("NB accuracy", old["report"]["accuracy"], new["report"]["accuracy"]),
        ("NB macro F1", old["report"]["macro"]["f1"], new["report"]["macro"]["f1"]),
        ("K-Means purity", baseline_km.purity, improved_km.purity),
        ("K-Means Rand index", baseline_km.rand_index, improved_km.rand_index),
        ("K-Means ARI", baseline_km.ari, improved_km.ari),
    ):
        rows.append(dict(metric=metric, baseline=before, improved=after, delta=after - before))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RES / "improvement_comparison.csv", index=False)
    print(comparison.to_string(index=False), flush=True)
    print("Paired classifier changes:", gains, "corrected,", losses, "regressed; exact p =", pvalue, flush=True)
    print("Clustering means across seeds:\n", km.groupby(["spherical", "init"])[["purity", "ari"]].mean(), flush=True)


if __name__ == "__main__":
    main()

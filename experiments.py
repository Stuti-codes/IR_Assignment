"""
Re-runs every experiment quoted in the report and saves the tables in results/
and the figures in figures/. Uses exactly the same code as the Streamlit app.

    python experiments.py            (about 1-2 minutes)
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_io import load_bundled, corpus_stats, noise_profile
from src.preprocess import PrepConfig, raw_tokens, vocab_stats, ensure_nltk
from src import pipeline as P
from src import metrics as M
from src.kmeans import top_terms_per_cluster
from src.recommender import ContentRecommender

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

COLORS = {"credit_card": "#4C78A8", "checking_savings": "#F58518", "money_transfer": "#E45756",
          "debt_collection": "#54A24B", "credit_reporting": "#72B7B2"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})

PRESETS = P.PRESETS
DEFAULT = PRESETS["D. B + lemmatisation (default)"]
QUERY = ("Someone pretending to be from my bank told me to send money through Zelle and now the bank "
         "says it was authorized and will not refund me")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


def averaged_scores(feats, docs, meta, labels, nb_features="counts", n_splits=5, km_seeds=(42, 7, 123), keep=None):
    """NB accuracy / macro-F1 averaged over several splits and K-Means averaged over several seeds,
    so small differences between settings are not just split / initialisation noise.
    `keep` fixes the documents (and therefore the splits) so settings are compared on the same data."""
    if keep is None:
        _, keep, _ = P.dedup_indices(docs, meta, feats["S"])
    lk = [labels[i] for i in keep]
    acc, f1 = [], []
    for sd in range(n_splits):
        a, b = P.stratified_split(lk, 0.2, sd)
        r = P.run_naive_bayes(feats["tokens"], feats["index"], labels, [keep[i] for i in a], [keep[i] for i in b], nb_features)
        acc.append(r["report"]["accuracy"])
        f1.append(r["report"]["macro"]["f1"])
    pur, ri, ari = [], [], []
    for sd in km_seeds:
        r = P.run_kmeans(feats["X"][keep], lk, 5, n_init=5, seed=sd)
        pur.append(r["purity"])
        ri.append(r["rand_index"])
        ari.append(r["ari"])
    pk = M.precision_at_k(feats["S"][np.ix_(keep, keep)], lk, (1, 5, 10))
    st, _ = vocab_stats(feats["tokens"])
    return {"docs_used": len(keep), "tokens": st["tokens"], "vocab_before_df": st["vocab"],
            "index_terms": len(feats["index"].vocab),
            "nb_acc_mean": np.mean(acc), "nb_acc_std": np.std(acc), "nb_macro_f1_mean": np.mean(f1),
            "km_purity_mean": np.mean(pur), "km_purity_std": np.std(pur), "km_ri_mean": np.mean(ri),
            "km_ari_mean": np.mean(ari), "p_at_1": pk[1], "p_at_5": pk[5], "p_at_10": pk[10]}


def topk_lists(S, seeds, k=10, exclude=None):
    out = []
    for i in seeds:
        s = S[i].copy()
        s[i] = -np.inf
        if exclude is not None:
            s[exclude[i]] = -np.inf
        out.append(set(np.argsort(-s)[:k].tolist()))
    return out


def main():
    t_all = time.perf_counter()
    print("nltk:", ensure_nltk()["notes"] or "ok")
    docs, meta = load_bundled()
    texts = docs["text"].tolist()
    labels = meta["category"].tolist()
    classes = sorted(set(labels))

    # ---------------------------------------------------------- 1. corpus
    cs = corpus_stats(docs, meta)
    pd.DataFrame([{k: v for k, v in cs.items() if k != "metadata_fields"}]).to_csv(
        os.path.join(RES, "corpus_stats.csv"), index=False)
    noise = noise_profile(texts)
    pd.DataFrame(list(noise.items()), columns=["noise", "docs"]).to_csv(os.path.join(RES, "noise_profile.csv"), index=False)
    print("corpus:", cs)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    vc = meta["category"].value_counts()
    ax[0].bar(vc.index, vc.values, color=[COLORS[c] for c in vc.index])
    for i, v in enumerate(vc.values):
        ax[0].text(i, v + 5, str(v), ha="center")
    ax[0].set_title("Documents per category")
    words = docs["text"].str.split().str.len()
    ax[1].boxplot([words[meta["category"] == c] for c in classes], showfliers=False)
    ax[1].set_xticks(range(1, len(classes) + 1), classes)
    ax[1].set_title("Words per complaint (outliers hidden)")
    save(fig, "fig01_corpus.png")

    # ---------------------------------------------------------- 2. default pipeline
    feats = P.build_features(texts, DEFAULT)
    idx, X, S = feats["index"], feats["X"], feats["S"]
    pd.DataFrame([idx.summary()]).to_csv(os.path.join(RES, "index_summary.csv"), index=False)
    pd.DataFrame([feats["timings"]]).to_csv(os.path.join(RES, "timings_features.csv"), index=False)

    raw_st, raw_tf = vocab_stats(raw_tokens(texts))
    new_st, new_tf = vocab_stats(feats["tokens"])
    ba = pd.DataFrame({"measure": list(raw_st.keys()), "before": list(raw_st.values()), "after": list(new_st.values())})
    ba.loc[len(ba)] = ["index_terms_after_df_filter", np.nan, len(idx.vocab)]
    ba.to_csv(os.path.join(RES, "vocab_before_after.csv"), index=False)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    tr = raw_tf.most_common(15)[::-1]
    ta = new_tf.most_common(15)[::-1]
    ax[0].barh([t for t, _ in tr], [c for _, c in tr], color="#4C78A8")
    ax[0].set_title("Top-15 terms BEFORE preprocessing")
    ax[1].barh([t for t, _ in ta], [c for _, c in ta], color="#54A24B")
    ax[1].set_title("Top-15 terms AFTER preprocessing")
    r = sorted(raw_tf.values(), reverse=True)
    a = sorted(new_tf.values(), reverse=True)
    ax[2].loglog(range(1, len(r) + 1), r, label=f"before (V={len(r):,})")
    ax[2].loglog(range(1, len(a) + 1), a, label=f"after (V={len(a):,})")
    ax[2].set_xlabel("rank")
    ax[2].set_ylabel("frequency")
    ax[2].set_title("Zipf plot")
    ax[2].legend()
    save(fig, "fig02_vocab_before_after.png")

    dfs = np.array([idx.df[t] for t in idx.vocab])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(dfs, bins=80, log=True, color="#4C78A8")
    ax[0].set_title("Document frequency of index terms")
    ax[0].set_xlabel("df")
    ax[1].hist([idx.idf[t] for t in idx.vocab], bins=40, color="#F58518")
    ax[1].set_title("IDF = log10(N/df)")
    save(fig, "fig03_df_idf.png")

    # duplicates
    pairs, keep, drop = P.dedup_indices(docs, meta, S, 0.85)
    pairs.drop(columns=["idx_a", "idx_b"]).to_csv(os.path.join(RES, "duplicate_pairs.csv"), index=False)
    print("duplicates:", pairs["type"].value_counts().to_dict(), "dropped", len(drop))
    n_title_diff = int(((pairs["type"] == "exact") & (pairs["title_a"] != pairs["title_b"])).sum())
    dup_summary = {
        "exact_pairs": int((pairs["type"] == "exact").sum()),
        "near_pairs": int((pairs["type"] == "near").sum()),
        "docs_removed": len(drop),
        "docs_kept": len(keep),
        "cross_label_pairs": int((~pairs["same_category"]).sum()),
        "exact_pairs_with_different_headline": n_title_diff,
        "near_min_jaccard": float(pairs.loc[pairs["type"] == "near", "jaccard_5shingle"].min()) if (pairs["type"] == "near").any() else None,
        "near_median_jaccard": float(pairs.loc[pairs["type"] == "near", "jaccard_5shingle"].median()) if (pairs["type"] == "near").any() else None,
        "removed_per_class": pd.Series([labels[i] for i in drop]).value_counts().to_dict(),
    }
    pd.DataFrame([dup_summary]).to_csv(os.path.join(RES, "duplicate_summary.csv"), index=False)

    # Naive Bayes on the de-duplicated corpus
    lk = [labels[i] for i in keep]
    trn, tst = P.stratified_split(lk, 0.2, 42)
    trn = [keep[i] for i in trn]
    tst = [keep[i] for i in tst]
    nb = P.run_naive_bayes(feats["tokens"], idx, labels, trn, tst)
    rep = nb["report"]
    pd.DataFrame(rep["per_class"]).to_csv(os.path.join(RES, "nb_per_class.csv"), index=False)
    pd.DataFrame(rep["confusion"], index=classes, columns=classes).to_csv(os.path.join(RES, "nb_confusion.csv"))
    pred = pd.DataFrame({"doc_id": meta["doc_id"].iloc[tst].values, "title": meta["title"].iloc[tst].values,
                         "actual": nb["y_true"], "predicted": nb["y_pred"]})
    pred.to_csv(os.path.join(RES, "nb_predictions.csv"), index=False)
    nb_top = {c: ", ".join(t for t, _ in nb["model"].top_terms(c, 12)) for c in classes}
    pd.DataFrame(list(nb_top.items()), columns=["class", "top terms"]).to_csv(os.path.join(RES, "nb_top_terms.csv"), index=False)
    summ_nb = {"accuracy": rep["accuracy"], **{f"macro_{k}": v for k, v in rep["macro"].items()},
               **{f"weighted_{k}": v for k, v in rep["weighted"].items()},
               "train": len(trn), "test": len(tst), **nb["timings"]}
    print("NB:", summ_nb)

    # NB stability over 5 splits
    accs = []
    for sd in range(5):
        a_, b_ = P.stratified_split(lk, 0.2, sd)
        r_ = P.run_naive_bayes(feats["tokens"], idx, labels, [keep[i] for i in a_], [keep[i] for i in b_])
        accs.append(r_["report"]["accuracy"])
    summ_nb["variant"] = "complement"
    summ_nb["features"] = "counts"
    summ_nb["alpha"] = 1.0
    summ_nb["acc_5_splits_mean"] = float(np.mean(accs))
    summ_nb["acc_5_splits_std"] = float(np.std(accs))
    pd.DataFrame([summ_nb]).to_csv(os.path.join(RES, "nb_summary.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cm = rep["confusion"]
    ax[0].imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[0].set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax[0].set_yticks(range(len(classes)), classes)
    ax[0].set_xlabel("predicted")
    ax[0].set_ylabel("actual")
    ax[0].set_title("Naive Bayes confusion matrix (test set)")
    w = 0.38
    xs = np.arange(len(classes))
    ax[1].bar(xs - w / 2, [nb["y_true"].count(c) for c in classes], w, label="actual")
    ax[1].bar(xs + w / 2, [nb["y_pred"].count(c) for c in classes], w, label="predicted")
    ax[1].set_xticks(xs, classes, rotation=30, ha="right")
    ax[1].set_ylim(0, max(nb["y_true"].count(c) for c in classes) * 1.25)
    ax[1].legend(ncol=2, loc="upper center")
    ax[1].set_title("Actual vs predicted class distribution")
    save(fig, "fig04_nb.png")

    # K-Means on the de-duplicated corpus
    km = P.run_kmeans(X[keep], lk, 5, n_init=5, seed=42)
    hist = pd.DataFrame(km["history"])
    hist.to_csv(os.path.join(RES, "kmeans_history.csv"), index=False)
    clusters = km["labels"].tolist()
    mem = pd.DataFrame({"doc_id": meta["doc_id"].iloc[keep].values, "title": meta["title"].iloc[keep].values,
                        "true_label": lk, "cluster": clusters,
                        "cluster_majority": [km["mapping"][c] for c in clusters]})
    mem.to_csv(os.path.join(RES, "kmeans_membership.csv"), index=False)
    ct = pd.crosstab(mem["cluster"], mem["true_label"])
    ct.to_csv(os.path.join(RES, "kmeans_contingency.csv"))
    tops = top_terms_per_cluster(km["centroids"], idx.vocab, 10)
    pd.DataFrame({"cluster": range(5), "majority": [km["mapping"][i] for i in range(5)],
                  "size": np.bincount(km["labels"], minlength=5),
                  "top_terms": [", ".join(t for t, _ in tt) for tt in tops]}).to_csv(
        os.path.join(RES, "kmeans_clusters.csv"), index=False)
    init_docs = pd.DataFrame({"centroid": range(5), "seed_doc": meta["doc_id"].iloc[[keep[i] for i in km["init_ids"]]].values,
                              "title": meta["title"].iloc[[keep[i] for i in km["init_ids"]]].values,
                              "label": [lk[i] for i in km["init_ids"]]})
    init_docs.to_csv(os.path.join(RES, "kmeans_init.csv"), index=False)
    summ_km = {"purity": km["purity"], "rand_index": km["rand_index"], "ari": km["ari"],
               "mapped_accuracy": km["mapped_accuracy"], "sse": km["sse"], "iterations": km["n_iter"],
               "stop_reason": km["stop_reason"], "restart_sse": ";".join(f"{x:.2f}" for x in km["all_runs_sse"]),
               **km["pairs"], "seconds": km["timings"]["k-means"],
               "spherical": km["spherical"], "init": "greedy k-means++", "n_init": 5}
    # stability of single runs
    pur = [P.run_kmeans(X[keep], lk, 5, n_init=1, seed=sd)["purity"] for sd in range(10)]
    summ_km["purity_single_run_min"] = min(pur)
    summ_km["purity_single_run_max"] = max(pur)
    summ_km["purity_single_run_mean"] = float(np.mean(pur))
    pd.DataFrame([summ_km]).to_csv(os.path.join(RES, "kmeans_summary.csv"), index=False)
    print("KM:", summ_km)

    # PCA through the gram matrix for the scatter (display only)
    K = S[np.ix_(keep, keep)].astype(np.float64)
    rm = K.mean(axis=1, keepdims=True)
    Kc = K - rm - rm.T + K.mean()
    vals, vecs = np.linalg.eigh(Kc)
    o = np.argsort(vals)[::-1][:2]
    coords = vecs[:, o] * np.sqrt(np.maximum(vals[o], 0))

    pal = ["#3969AC", "#F2B701", "#E73F74", "#11A579", "#7F3C8D"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for c in range(5):
        m_ = np.array(clusters) == c
        ax[0].scatter(coords[m_, 0], coords[m_, 1], s=6, alpha=0.6, color=pal[c], label=f"C{c} ({km['mapping'][c]})")
        ax[0].scatter(coords[m_, 0].mean(), coords[m_, 1].mean(), marker="X", s=120, color="black")
    ax[0].legend(markerscale=3, fontsize=8)
    ax[0].set_title("K-Means clusters in 2-D (PCA, display only)")
    sizes = np.bincount(clusters, minlength=5)
    ax[1].bar(range(5), sizes, color=pal)
    ax[1].set_xticks(range(5), [f"C{c} {km['mapping'][c]}" for c in range(5)], rotation=30, ha="right")
    for i, v in enumerate(sizes):
        ax[1].text(i, v + 5, str(v), ha="center")
    ax[1].set_title("Documents per cluster")
    ax[2].imshow(ct.values, cmap="Greens")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax[2].text(j, i, ct.values[i, j], ha="center", va="center", fontsize=8)
    ax[2].set_xticks(range(ct.shape[1]), ct.columns, rotation=30, ha="right")
    ax[2].set_yticks(range(ct.shape[0]), [f"C{i}" for i in ct.index])
    ax[2].set_title("Cluster x true label")
    save(fig, "fig05_kmeans.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(hist["iteration"], hist["sse"], marker="o")
    ax2 = ax.twinx()
    ax2.bar(hist["iteration"], hist["docs_moved"], alpha=0.3, color="gray")
    ax.set_xlabel("iteration")
    ax.set_ylabel("SSE")
    ax2.set_ylabel("docs re-assigned")
    ax.set_title("K-Means convergence (best of 5 restarts)")
    save(fig, "fig06_kmeans_convergence.png")

    # comparison
    comp = pd.DataFrame([
        ["Naive Bayes", "Accuracy", rep["accuracy"]], ["Naive Bayes", "Macro precision", rep["macro"]["precision"]],
        ["Naive Bayes", "Macro recall", rep["macro"]["recall"]], ["Naive Bayes", "Macro F1", rep["macro"]["f1"]],
        ["K-Means", "Purity", km["purity"]], ["K-Means", "Rand index", km["rand_index"]],
        ["K-Means", "Adjusted RI", km["ari"]], ["K-Means", "Majority-label acc. (= purity)", km["mapped_accuracy"]],
    ], columns=["method", "metric", "value"])
    comp.to_csv(os.path.join(RES, "nb_vs_kmeans.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = ["#4C78A8" if m_ == "Naive Bayes" else "#F58518" for m_ in comp["method"]]
    ax.bar(comp["metric"], comp["value"], color=colors)
    for i, v in enumerate(comp["value"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(comp)), comp["metric"], rotation=25, ha="right")
    ax.set_title("Naive Bayes (blue) vs K-Means (orange)")
    save(fig, "fig07_comparison.png")

    # ---------------------------------------------------------- 3. recommendations
    rec = ContentRecommender(idx, docs, meta)
    examples = []
    # an impersonation Zelle scam, a card fraud dispute, and a credit-reporting template letter that has copies
    template = pairs[(pairs["type"] == "exact") & pairs["doc_a"].isin(meta.loc[meta["category"] == "credit_reporting", "doc_id"])]
    seeds_rec = ["CFPB-5156048", "CFPB-4570909", template["doc_a"].iloc[0]]
    for did in seeds_rec:
        d = int(np.where(docs["doc_id"] == did)[0][0])
        for hide in (False, True):
            res = rec.similar_to_doc(d, 5, hide_dups=hide, dup_threshold=0.9)
            res.insert(0, "seed_title", meta["title"].iloc[d])
            res.insert(0, "seed", did)
            res.insert(2, "dup_filter", hide)
            examples.append(res.drop(columns=["idx", "snippet"]))
    pd.concat(examples).to_csv(os.path.join(RES, "recommendation_examples.csv"), index=False)
    q_tokens = P.preprocess_corpus([QUERY], DEFAULT)[0]
    qres, _ = rec.similar_to_tokens(q_tokens, 5)
    qres.drop(columns=["idx", "snippet"]).to_csv(os.path.join(RES, "recommendation_query_example.csv"), index=False)

    ks = list(range(1, 21))
    excl = S >= 0.9
    np.fill_diagonal(excl, False)
    p_all = M.precision_at_k(S, labels, ks)
    p_flt = M.precision_at_k(S, labels, ks, exclude=excl)
    pk = pd.DataFrame({"K": ks, "all_candidates": [p_all[k] for k in ks], "near_dups_removed": [p_flt[k] for k in ks]})
    pk.to_csv(os.path.join(RES, "precision_at_k.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ks, pk["all_candidates"], marker="o", ms=3, label="all candidates")
    ax.plot(ks, pk["near_dups_removed"], marker="o", ms=3, label="near-duplicates (cos >= 0.9) removed")
    ax.set_xlabel("K")
    ax.set_ylabel("precision@K (same category)")
    ax.set_title("Recommendation precision@K")
    ax.legend()
    save(fig, "fig08_precision_at_k.png")

    # ---------------------------------------------------------- 4. preprocessing ablation
    base_feats = feats
    rng = np.random.default_rng(0)
    seeds = rng.choice(len(docs), 300, replace=False)
    base_lists = topk_lists(base_feats["S"], seeds)
    abl = []
    variants = [(name, cfg, "log", "idf", "counts") for name, cfg in PRESETS.items()]
    # weighting variants on the default preprocessing; NB uses tf-idf features there so the weighting matters
    variants += [("H. D, NB on tf-idf features", DEFAULT, "log", "idf", "tfidf"),
                 ("I. D, raw tf (NB on tf-idf)", DEFAULT, "raw", "idf", "tfidf"),
                 ("J. D, no idf (NB on tf-idf)", DEFAULT, "log", "none", "tfidf")]
    for name, cfg, tf_s, idf_s, nbf in variants:
        t0 = time.perf_counter()
        if cfg == DEFAULT and tf_s == "log" and idf_s == "idf":
            f_ = base_feats
        else:
            f_ = P.build_features(texts, cfg, 2, 0.9, tf_s, idf_s)
        # same de-duplicated documents and splits for every setting, so only the preprocessing changes
        r_ = averaged_scores(f_, docs, meta, labels, nbf, keep=keep)
        lists = topk_lists(f_["S"], seeds)
        r_["top10_overlap_with_D"] = np.mean([len(x & y) / 10 for x, y in zip(lists, base_lists)])
        r_["setting"] = name
        r_["config"] = cfg.short_name()
        r_["tf"], r_["idf"], r_["nb_features"] = tf_s, idf_s, nbf
        r_["seconds"] = time.perf_counter() - t0
        abl.append(r_)
        print("ablation %-38s V=%6d NB=%.4f+-%.4f KMpur=%.4f+-%.4f ARI=%.4f P@10=%.4f ov=%.2f" % (
            name, r_["index_terms"], r_["nb_acc_mean"], r_["nb_acc_std"], r_["km_purity_mean"],
            r_["km_purity_std"], r_["km_ari_mean"], r_["p_at_10"], r_["top10_overlap_with_D"]))
    abl = pd.DataFrame(abl)
    abl.to_csv(os.path.join(RES, "preprocessing_ablation.csv"), index=False)

    a6 = abl.iloc[:len(PRESETS)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    short = [n.split(".")[0] for n in a6["setting"]]
    xs = np.arange(len(a6))
    ax[0].bar(xs - 0.2, a6["vocab_before_df"], 0.4, color="#BAB0AC", label="vocabulary after preprocessing")
    ax[0].bar(xs + 0.2, a6["index_terms"], 0.4, color="#72B7B2", label="kept in index (min_df=2, max_df=0.9)")
    ax[0].set_xticks(xs, short)
    ax[0].set_yscale("log")
    ax[0].set_title("Vocabulary size per setting (log scale)")
    ax[0].legend(fontsize=7)
    xs = np.arange(len(a6))
    w = 0.2
    for j, (col, lab) in enumerate((("nb_acc_mean", "NB accuracy (5 splits)"), ("km_purity_mean", "K-Means purity (3 seeds)"),
                                    ("km_ari_mean", "K-Means ARI (3 seeds)"), ("p_at_10", "P@10"))):
        ax[1].bar(xs + (j - 1.5) * w, a6[col], w, label=lab)
    ax[1].set_xticks(xs, short)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].legend(ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax[1].set_title("Model quality per setting (A-G, see table)")
    save(fig, "fig09_ablation.png")

    # ---------------------------------------------------------- 5. duplicate impact
    texts_norm = docs["text"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip().tolist()
    di = []
    for name, use in (("with duplicates", list(range(len(docs)))), ("de-duplicated", keep)):
        lu = [labels[i] for i in use]
        accs, leak, leak_correct, clean_correct, clean_n = [], [], 0, 0, 0
        for sd in range(10):
            a_, b_ = P.stratified_split(lu, 0.2, sd)
            a_ = [use[i] for i in a_]
            b_ = [use[i] for i in b_]
            r_ = P.run_naive_bayes(feats["tokens"], idx, labels, a_, b_)
            accs.append(r_["report"]["accuracy"])
            tr_set = set(texts_norm[i] for i in a_)
            leaked = [texts_norm[i] in tr_set for i in b_]
            leak.append(sum(leaked))
            for is_leak, yt, yp in zip(leaked, r_["y_true"], r_["y_pred"]):
                if is_leak:
                    leak_correct += yt == yp
                else:
                    clean_correct += yt == yp
                    clean_n += 1
        kms_ = [P.run_kmeans(X[use], lu, 5, n_init=5, seed=sd) for sd in (42, 7, 123)]
        Su = S[np.ix_(use, use)]
        pk_ = M.precision_at_k(Su, lu, (1, 5, 10))
        Su2 = Su.copy()
        np.fill_diagonal(Su2, -1)
        di.append({"corpus": name, "docs": len(use), "nb_acc_mean_10_splits": np.mean(accs),
                   "nb_acc_std": np.std(accs), "test_docs_with_exact_copy_in_train": np.mean(leak),
                   "nb_acc_on_leaked_test_docs": leak_correct / sum(leak) if sum(leak) else np.nan,
                   "nb_acc_on_clean_test_docs": clean_correct / clean_n,
                   "km_purity_mean_3_seeds": np.mean([k["purity"] for k in kms_]),
                   "km_purity_values": ";".join("%.4f" % k["purity"] for k in kms_),
                   "km_ri_mean": np.mean([k["rand_index"] for k in kms_]),
                   "km_ari_mean": np.mean([k["ari"] for k in kms_]),
                   "p_at_1": pk_[1], "p_at_5": pk_[5], "p_at_10": pk_[10],
                   "top1_rec_is_duplicate": float((Su2.max(axis=1) >= 0.9).mean())})
    pd.DataFrame(di).to_csv(os.path.join(RES, "duplicate_impact.csv"), index=False)
    print(pd.DataFrame(di).round(4).to_string())

    # ---------------------------------------------------------- 6. k sweep
    sweep = []
    for k in range(2, 13):
        r_ = P.run_kmeans(X[keep], lk, k, n_init=3)
        sweep.append({"k": k, "sse": r_["sse"], "purity": r_["purity"], "rand_index": r_["rand_index"], "ari": r_["ari"]})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(os.path.join(RES, "k_sweep.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(sweep["k"], sweep["sse"], marker="o")
    ax[0].set_title("Elbow: SSE vs k")
    for col in ("purity", "rand_index", "ari"):
        ax[1].plot(sweep["k"], sweep[col], marker="o", label=col)
    ax[1].axvline(5, ls="--", color="gray")
    ax[1].legend()
    ax[1].set_title("External measures vs k")
    save(fig, "fig10_k_sweep.png")

    print("done in %.1f s" % (time.perf_counter() - t_all))


if __name__ == "__main__":
    main()

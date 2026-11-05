"""
A few extra tables used in the report that experiments.py does not produce:

  tfidf_worked_example.csv     one ltc weight worked out by hand, checked against the index
  same_test_comparison.csv     NB and K-Means scored on the same 736 test complaints
  nb_confidence.csv            how confident Complement NB is when it is right and when it is wrong
  dup_threshold_sensitivity.csv  K-Means and recommendation quality for several near-duplicate cut-offs

Run after experiments.py:   python report_extras.py   (about one minute)
"""

import math
import os

import numpy as np
import pandas as pd

from src import metrics as M
from src import pipeline as P
from src.data_io import docs_to_drop, find_duplicates, load_bundled
from src.naive_bayes import to_count_dicts

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT = P.PRESETS["D. B + lemmatisation (default)"]


def worked_example(feats, docs, meta, term="zelle"):
    idx = feats["index"]
    # the complaint where the term is used most often
    d, tf = max(idx.postings[term], key=lambda p: p[1])
    tf_w = 1 + math.log10(tf)
    idf = math.log10(idx.N / idx.df[term])
    raw = tf_w * idf
    length = idx.doc_lengths[d]
    return pd.DataFrame([{
        "doc_id": docs["doc_id"].iloc[d], "title": meta["title"].iloc[d], "term": term,
        "tf": tf, "df": idx.df[term], "N": idx.N, "tf_weight": tf_w, "idf": idf,
        "raw_weight": raw, "doc_vector_length": length, "normalised_weight": raw / length,
        "weight_in_index": idx.doc_vectors[d][term],
        "doc_terms": len(idx.doc_vectors[d]),
    }])


def same_test_comparison(meta):
    pred = pd.read_csv(os.path.join(RES, "nb_predictions.csv"))
    memb = pd.read_csv(os.path.join(RES, "kmeans_membership.csv"))
    rows = []
    rep = M.classification_report(pred["actual"].tolist(), pred["predicted"].tolist(),
                                  sorted(meta["category"].unique()))
    rows.append({"method": "Naive Bayes (predicted class used as a cluster)", "documents": len(pred),
                 "accuracy": rep["accuracy"], "macro_f1": rep["macro"]["f1"],
                 "purity": M.purity(pred["predicted"].tolist(), pred["actual"].tolist()),
                 "rand_index": M.rand_index(pred["predicted"].tolist(), pred["actual"].tolist())})
    sub = memb[memb["doc_id"].isin(pred["doc_id"])]
    rows.append({"method": "K-Means, same test complaints", "documents": len(sub),
                 "accuracy": float((sub["cluster_majority"] == sub["true_label"]).mean()), "macro_f1": np.nan,
                 "purity": M.purity(sub["cluster"].tolist(), sub["true_label"].tolist()),
                 "rand_index": M.rand_index(sub["cluster"].tolist(), sub["true_label"].tolist())})
    rows.append({"method": "K-Means, all de-duplicated complaints", "documents": len(memb),
                 "accuracy": float((memb["cluster_majority"] == memb["true_label"]).mean()), "macro_f1": np.nan,
                 "purity": M.purity(memb["cluster"].tolist(), memb["true_label"].tolist()),
                 "rand_index": M.rand_index(memb["cluster"].tolist(), memb["true_label"].tolist())})
    return pd.DataFrame(rows)


def nb_confidence(feats, docs, meta, keep):
    labels = meta["category"].tolist()
    lk = [labels[i] for i in keep]
    tr, te = P.stratified_split(lk, 0.2, 42)
    nb = P.run_naive_bayes(feats["tokens"], feats["index"], labels, [keep[i] for i in tr], [keep[i] for i in te])
    vecs = to_count_dicts([feats["tokens"][keep[i]] for i in te])
    top = [max(nb["model"].predict_proba(v).values()) for v in vecs]
    ok = np.array(nb["y_true"]) == np.array(nb["y_pred"])
    top = np.array(top)
    return pd.DataFrame([{
        "test_docs": len(ok), "errors": int((~ok).sum()),
        "errors_with_score_ge_0.99": int(((~ok) & (top >= 0.99)).sum()),
        "correct_with_score_ge_0.99": int((ok & (top >= 0.99)).sum()),
        "median_score_correct": float(np.median(top[ok])), "median_score_wrong": float(np.median(top[~ok])),
        "accuracy_score_ge_0.99": float(ok[top >= 0.99].mean()),
        "accuracy_score_lt_0.9": float(ok[top < 0.9].mean()) if (top < 0.9).any() else np.nan,
        "share_score_lt_0.9": float((top < 0.9).mean()),
    }])


def threshold_sensitivity(feats, docs, meta):
    labels = np.array(meta["category"].tolist())
    S, X = feats["S"], feats["X"]
    rows = []
    for thr in (0.80, 0.85, 0.90, 0.95):
        pairs = find_duplicates(docs, meta, S, thr)
        drop = set(docs_to_drop(pairs))
        keep = [i for i in range(len(docs)) if i not in drop]
        km = P.run_kmeans(X[keep], labels[keep].tolist(), 5)
        Sk = S[np.ix_(keep, keep)]
        pk = M.precision_at_k(Sk, labels[keep].tolist(), (5, 10))
        rows.append({"cosine_threshold": thr, "near_pairs": int((pairs["type"] == "near").sum()),
                     "docs_removed": len(drop), "docs_kept": len(keep), "km_purity": km["purity"],
                     "km_rand_index": km["rand_index"], "km_ari": km["ari"], "p_at_5": pk[5], "p_at_10": pk[10]})
    return pd.DataFrame(rows)


def main():
    docs, meta = load_bundled()
    feats = P.build_features(docs["text"].tolist(), DEFAULT)
    _, keep, _ = P.dedup_indices(docs, meta, feats["S"])

    worked_example(feats, docs, meta).to_csv(os.path.join(RES, "tfidf_worked_example.csv"), index=False)
    same_test_comparison(meta).to_csv(os.path.join(RES, "same_test_comparison.csv"), index=False)
    nb_confidence(feats, docs, meta, keep).to_csv(os.path.join(RES, "nb_confidence.csv"), index=False)
    threshold_sensitivity(feats, docs, meta).to_csv(os.path.join(RES, "dup_threshold_sensitivity.csv"), index=False)
    for f in ("tfidf_worked_example", "same_test_comparison", "nb_confidence", "dup_threshold_sensitivity"):
        print(f)
        print(pd.read_csv(os.path.join(RES, f + ".csv")).round(4).to_string(index=False), "\n")


if __name__ == "__main__":
    main()

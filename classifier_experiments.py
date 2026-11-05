"""Evaluate the SVM ensemble: python classifier_experiments.py."""
from pathlib import Path
import numpy as np
import pandas as pd
from src import pipeline as P
from src.classifier import run_classifier
from src.data_io import load_bundled


def main():
    results = Path(__file__).resolve().parent / "results"
    docs, meta = load_bundled()
    texts, labels = docs.text.tolist(), meta.category.tolist()
    feats = P.build_features(texts, P.PRESETS["D. B + lemmatisation (default)"])
    _, keep, _ = P.dedup_indices(docs, meta, feats["S"])
    kept_labels = [labels[i] for i in keep]
    stability = []
    for seed in (42, 0, 1, 2, 3, 4):
        a, b = P.stratified_split(kept_labels, .2, seed)
        train, test = [keep[i] for i in a], [keep[i] for i in b]
        result = run_classifier(texts, labels, train, test)
        report = result["report"]
        stability.append(dict(seed=seed, accuracy=report["accuracy"], macro_f1=report["macro"]["f1"]))
        print("seed", seed, stability[-1], flush=True)
        if seed == 42:
            validation = []
            for inner_seed in (0, 1, 2):
                ia, ib = P.stratified_split([labels[i] for i in train], .2, inner_seed)
                inner = run_classifier(texts, labels, [train[i] for i in ia], [train[i] for i in ib])["report"]
                validation.append(dict(seed=inner_seed, accuracy=inner["accuracy"], macro_f1=inner["macro"]["f1"]))
            pd.DataFrame(validation).to_csv(results / "classifier_validation.csv", index=False)
            original = pd.read_csv(results / "baseline" / "nb_predictions.csv")
            assert original.doc_id.astype(str).tolist() == meta.doc_id.iloc[test].astype(str).tolist()
            pred = meta.iloc[test][["doc_id", "title"]].copy()
            pred["actual"], pred["predicted"] = result["y_true"], result["y_pred"]
            pred.to_csv(results / "classifier_predictions.csv", index=False)
            pd.DataFrame(report["confusion"], index=result["classes"], columns=result["classes"]).to_csv(
                results / "classifier_confusion.csv")
            pd.DataFrame(report["per_class"]).to_csv(results / "classifier_per_class.csv", index=False)
            summary = dict(model="two-view linear SVM", accuracy=report["accuracy"], macro_f1=report["macro"]["f1"],
                           train=len(train), test=len(test))
    table = pd.DataFrame(stability)
    table.to_csv(results / "classifier_split_stability.csv", index=False)
    extra = table[table.seed != 42]
    summary.update(repeated_holdout_mean=extra.accuracy.mean(), repeated_holdout_std=extra.accuracy.std(ddof=0))
    pd.DataFrame([summary]).to_csv(results / "classifier_summary.csv", index=False)
    print("Summary", summary, flush=True)


if __name__ == "__main__":
    main()

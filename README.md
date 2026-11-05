# Complaint & Fraud Ops Workbench
IR Assignment 2 - Text Mining & Document Intelligence System (Streamlit)

Context: the complaints and fraud operations team of a retail bank has to route
free-text customer complaints to the right product team, find recurring problems
and scam patterns, recognise mass-submitted template complaints and pull up
similar past cases. Dataset: 4,000 real complaint narratives from the CFPB
Consumer Complaint Database (Jul 2021 - Dec 2022), 800 for each of five products:
credit card, checking/savings account, money transfer, debt collection, credit reporting.

## 1. Setup

Python 3.9 or newer.

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

The NLTK download is a one-time step. If the lab machine has no internet the
app still runs: it falls back to a built-in stop-word list and uses the
Porter stemmer instead of the WordNet lemmatiser (a warning is shown under the
pipeline status bar).

## 2. Run the application

```bash
python -m streamlit run app.py
```

(`streamlit run app.py` also works if the `streamlit` command is on the PATH.)

Then open the URL printed in the terminal (normally http://localhost:8501).
The bundled corpus is loaded automatically; the first page load takes a few
seconds while the index is built.

The six pages are in the navigation bar at the top, in the order of the six
workflow parts of the assignment brief:

| Navigation label | Page (brief item) |
| --- | --- |
| Dashboard | 1. Dashboard and corpus overview |
| Data ingestion | 2. Data upload / ingestion |
| Preprocessing | 3. Text preprocessing configuration (incl. inverted index, TF-IDF, cosine) |
| Text mining hub | 4. Text mining and corpus analytics hub (NB, K-Means, Purity / RI, insights, extra SVM tab) |
| Recommendations | 5. Document recommendation panel |
| Performance analytics | 6. Performance analytics |

The app opens in the light theme. For the dark theme use the menu at the top
right -> Settings -> Theme (both themes are defined in `.streamlit/config.toml`).

On the BITS Virtual Lab, if the browser runs on a different machine, use:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Windows CMD note: to change to a folder on another drive use `cd /d "D:\path\with spaces"`.

## 3. Reproduce the experiments

```bash
python experiments.py
```

Writes all tables to `results/` and figures to `figures/` (about 2 minutes).

For baseline comparisons, SVM evaluation and tests:

```bash
python evaluate_models.py
python classifier_experiments.py
python report_extras.py
python -m unittest discover -s tests -v
```

`report_extras.py` adds four small tables used in the report: a worked TF-IDF
weight, NB and K-Means on the same test complaints, NB confidence, and the
near-duplicate threshold sensitivity. Two of the unit tests reproduce worked
examples from Manning et al. (Fig. 16.4 Purity / Rand Index, Example 13.1 Naive Bayes).

The default classifier is manual Complement Naive Bayes with counts and
alpha=1. The mining page also offers Multinomial Naive Bayes.
Classification TF-IDF learns its vocabulary and IDF from the training split only.
The default clustering uses manual spherical K-Means with greedy k-means++
initialisation and five restarts; turn off spherical mode and select k-means++
to reproduce the original Euclidean algorithm. Restarts are selected by minimum
objective, without class labels. SSE values from the two objectives cannot be
compared directly.

Reference NB predictions are in `results/baseline/nb_predictions.csv` and are
used to verify the comparison scripts. Methods, results and
evaluation limits are documented in `results/EXPERIMENTS.md`; the report
(`G23_IR_Assignment2_Report.pdf`) uses the same CSV results. Secondary numbers
(ablation, stability) can differ in the third decimal place on another machine
because of floating-point ties; the headline numbers are reproducible.

The **Extra - SVM benchmark** tab is beyond the brief and provides a second classification method.
It averages two linear SVMs with training-fitted manual TF-IDF: lemmatised
unigrams/bigrams/trigrams and original-form unigrams/bigrams. The model is
reported separately from the required manual Naive Bayes. Its fixed seed-42
test accuracy is 79.35%, with 80.82% mean accuracy over five additional splits;
see `results/EXPERIMENTS.md` and `results/classifier_summary.csv` for details.

`python nb_parameter_search.py` reproduces the training-only NB parameter
search (Bernoulli, multinomial, complement, one-vs-rest and chi-square selection).

`prepare_data.py` documents how the corpus was built. The CFPB stopped
publishing complaint narratives in August 2026, so the text comes from an
archived copy of the database on Hugging Face
(https://huggingface.co/datasets/Mouwiya/cfpb-consumer-complaints, parquet
shards 0059-0065). `python prepare_data.py --from-parquet <folder>` rebuilds
`data/raw/cfpb_complaints_sample.csv` from those shards; `python prepare_data.py`
on its own only splits that file into `documents.csv` and `metadata.csv`.

## 4. Project structure

```
app.py                  Streamlit entry point, top navigation, pipeline status bar
.streamlit/config.toml  light and dark theme
assets/                 logo and page icon
views/
  common.py             session state, caching, shared helpers
  dashboard.py          1. dashboard & corpus overview
  ingestion.py          2. upload / ingestion, corpus stats, duplicate detection
  preprocessing.py      3. preprocessing config, before/after stats, index explorer
  mining.py             4. Naive Bayes, K-Means, evaluation & comparison, insights, SVM
  recommend.py          5. Top-K recommendation panel
  performance.py        6. timings, precision@K, preprocessing / duplicate / k experiments
src/
  preprocess.py         cleaning, XXXX redaction removal, tokenisation, stop-words, stemming, lemmatisation, n-grams
  tfidf_index.py        inverted index, tf-idf (ltc) with cosine normalisation, cosine scoring
  naive_bayes.py        multinomial and complement Naive Bayes (manual)
  classifier.py         word N-gram SVM ensemble
  kmeans.py             Euclidean / spherical K-Means and greedy k-means++ (manual)
  metrics.py            accuracy, P/R/F1, confusion matrix, purity, Rand index, ARI, precision@K
  recommender.py        content-based Top-K recommender
  data_io.py            CSV / JSON / ZIP loading, text-metadata split, noise profile, duplicate detection
  pipeline.py           glue used by both the app and experiments.py (incl. preprocessing presets)
data/
  documents.csv         doc_id, text                      (complaint narrative only)
  metadata.csv          doc_id, title, category, product, issue, company, state, date, fraud flag, ...
  raw/cfpb_complaints_sample.csv   the sampled raw records
  samples/              small JSON and ZIP files to try the upload page
experiments.py          reproduces every table / figure used in the report
evaluate_models.py      validation, paired baseline and stability comparisons
report_extras.py        worked TF-IDF example, same-test comparison, NB confidence, duplicate threshold
classifier_experiments.py  SVM reference test and repeated-split evaluation
nb_parameter_search.py  NB feature and parameter search
prepare_data.py         CFPB parquet shards -> sample -> documents.csv + metadata.csv
results/, figures/      output of experiments.py
virtual_lab/            screenshots (and screen recording) of the run in the BITS Virtual Lab
tests/                  unit tests for the algorithms, the SVM and every app page
```

## 5. What is implemented by hand

TF-IDF and the inverted index, cosine similarity, Naive Bayes, K-Means,
Purity and Rand Index are written from scratch (no TfidfVectorizer,
sklearn.naive_bayes, sklearn.cluster or sklearn.metrics).
Libraries are used for: NLTK stop-word list / Porter stemmer / WordNet
lemmatiser, scipy to store the computed weights as a sparse matrix and for the
2-D PCA projection, plotly and matplotlib for charts, and scikit-learn's t-SNE
as an optional 2-D projection for the cluster scatter plot (visualisation only).
The SVM ensemble uses scikit-learn's LinearSVC. The required Naive Bayes
implementation remains manual.

## 6. Uploading another dataset

The Data ingestion page accepts:
- CSV or JSON with one row per document (choose the id, text, label and title columns)
- ZIP of .txt files, one file per document; the parent folder name is used as the label

Try `data/samples/cfpb_sample_100.json` or `data/samples/cfpb_sample_txt.zip`.

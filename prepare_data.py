"""
Builds the corpus used by the app from the CFPB Consumer Complaint Database.

Two steps:

1) (only needed once, already done - the result is data/raw/cfpb_complaints_sample.csv)
   The CFPB stopped publishing complaint narratives in August 2026, so the
   text comes from an archived full copy of the database on Hugging Face:
       https://huggingface.co/datasets/Mouwiya/cfpb-consumer-complaints
   We downloaded the parquet shards 0059 - 0065 (complaints received
   May 2020 - Feb 2023) from
       .../resolve/refs%2Fconvert%2Fparquet/default/train/00NN.parquet
   and took a stratified random sample:
       - 5 products that matter for a bank's complaints / fraud team
       - only complaints with a narrative of at least 40 words
       - received between 2021-07-01 and 2022-12-31 (same product taxonomy)
       - 800 complaints per product, random_state=26
   Run:  python prepare_data.py --from-parquet <folder with the .parquet files>

2) Split the raw sample into the two files the app reads:
       data/documents.csv  -> doc_id, text                    (narrative only)
       data/metadata.csv   -> doc_id, title, category, issue, company, ... (everything else)
   Run:  python prepare_data.py

The narratives are kept exactly as published (XXXX redactions, {$100.00}
amounts, ALL CAPS, typos). Cleaning happens in the preprocessing step of the
app so its effect can be measured.
"""

import argparse
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "cfpb_complaints_sample.csv")
OUT_DIR = os.path.join(HERE, "data")

# full CFPB product name -> short label used in the app
PRODUCTS = {
    "Credit card or prepaid card": "credit_card",
    "Checking or savings account": "checking_savings",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Debt collection": "debt_collection",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
}

# issue / sub_issue wording that points to fraud, scams or identity theft
FRAUD_WORDS = ("fraud|scam|identity theft|unauthori|stolen|not mine|didn't make|did not make|"
               "belongs to someone else|don't recognize|do not recognize")


def build_sample(parquet_dir, per_product=800, seed=26):
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise SystemExit("no parquet files in " + parquet_dir)
    cols = ["complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue",
            "company", "state", "tags", "submitted_via", "company_response_to_consumer",
            "consumer_complaint_narrative"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    print("rows in shards:", len(df))

    df = df[df["product"].isin(PRODUCTS)]
    df = df[df["consumer_complaint_narrative"].fillna("").str.len() > 0]
    df = df[(df["date_received"] >= "2021-07-01") & (df["date_received"] < "2023-01-01")]
    words = df["consumer_complaint_narrative"].str.count(" ") + 1
    df = df[words >= 40]
    print("candidates:", len(df))

    sample = pd.concat([g.sample(per_product, random_state=seed) for _, g in df.groupby("product")])
    sample = sample.sort_values("date_received").reset_index(drop=True)
    sample["date_received"] = sample["date_received"].dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    sample.to_csv(RAW, index=False, encoding="utf-8")
    print("saved", RAW, sample.shape)


def make_title(row):
    # complaints have no headline, so issue + company is used as a short title
    company = str(row["company"]).title()
    company = company if len(company) <= 35 else company[:33] + ".."
    return "%s - %s" % (row["issue"], company)


def split():
    raw = pd.read_csv(RAW, dtype={"complaint_id": str})
    print("raw sample:", raw.shape)

    ids = "CFPB-" + raw["complaint_id"].astype(str)
    docs = pd.DataFrame({"doc_id": ids, "text": raw["consumer_complaint_narrative"]})

    text_fields = (raw["issue"].fillna("") + " " + raw["sub_issue"].fillna("")).str.lower()
    meta = pd.DataFrame({
        "doc_id": ids,
        "title": raw.apply(make_title, axis=1),
        "category": raw["product"].map(PRODUCTS),
        "product": raw["product"],
        "sub_product": raw["sub_product"],
        "issue": raw["issue"],
        "sub_issue": raw["sub_issue"],
        "company": raw["company"],
        "state": raw["state"],
        "date_received": raw["date_received"],
        "tags": raw["tags"],
        "fraud_related": text_fields.str.contains(FRAUD_WORDS, regex=True),
        "source": "CFPB Consumer Complaint Database (archived snapshot)",
    })

    docs.to_csv(os.path.join(OUT_DIR, "documents.csv"), index=False, encoding="utf-8")
    meta.to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False, encoding="utf-8")
    print("documents.csv:", docs.shape, " metadata.csv:", meta.shape)
    print(meta["category"].value_counts().to_string())
    print("fraud related:", int(meta["fraud_related"].sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-parquet", help="folder with the downloaded parquet shards")
    args = ap.parse_args()
    if args.from_parquet:
        build_sample(args.from_parquet)
    split()

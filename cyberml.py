"""


Tasks:
  url          : URL features + Logistic Regression (classification)
  email        : Email TF-IDF + RandomForest (classification)
  email-kmeans : Email TF-IDF + KMeans (unsupervised bonus)

Examples:
  python cyberml.py url          --csv url_spam_classification.csv --url-col url --label-col is_spam
  python cyberml.py email        --csv emails.csv --text-col text --label-col spam
  python cyberml.py email-kmeans --csv emails.csv --text-col text --k 2
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

RANDOM_SEED = 42


# ---------------------- plotting helpers ----------------------
def ensure_reports_dir():
    Path("reports").mkdir(exist_ok=True)

def save_roc_pr(y_true, y_prob, title_prefix, stem):
    ensure_reports_dir()
    # ROC
    fig1, ax1 = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax1)
    ax1.set_title(f"{title_prefix}: ROC Curve")
    fig1.tight_layout()
    fig1.savefig(f"reports/{stem}_roc.png", dpi=150)
    plt.close(fig1)
    # PR
    fig2, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax2)
    ax2.set_title(f"{title_prefix}: Precision–Recall Curve")
    fig2.tight_layout()
    fig2.savefig(f"reports/{stem}_pr.png", dpi=150)
    plt.close(fig2)

def print_cls_metrics(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"ROC AUC  : {auc:.4f}")
        except Exception:
            pass
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred), "\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))


# ---------------------- robust loaders ----------------------
def load_email_csv_robust(csv_path: Path, text_col_hint="text", label_col_hint="spam") -> pd.DataFrame:
    """
    For emails.csv that has many trailing commas after the first two columns.
    Read only first two columns and tidy the text/labels.
    """
    try:
        df = pd.read_csv(
            csv_path,
            header=0,
            usecols=[0, 1],          # only first two columns
            engine="python",
            on_bad_lines="skip"
        )
    except Exception:
        df_all = pd.read_csv(csv_path, header=0, engine="python", on_bad_lines="skip")
        df = df_all.iloc[:, :2].copy()

    # normalise column names to [text, spam]
    if len(df.columns) < 2:
        raise ValueError("Expected at least 2 columns (text + spam).")
    df.columns = ["text", "spam"]

    # tidy text
    df["text"] = (
        df["text"].astype(str)
        .str.replace(r'^\s*"?Subject:\s*', "", flags=re.IGNORECASE, regex=True)
        .str.replace(r'^"|"$', "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # labels: strings -> 0/1
    if df["spam"].dtype == object:
        mapping = {"spam": 1, "phish": 1, "malicious": 1, "ham": 0, "benign": 0, "safe": 0}
        df["spam"] = df["spam"].astype(str).str.lower().map(mapping)

    df["spam"] = pd.to_numeric(df["spam"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["text", "spam"]).copy()
    df["spam"] = df["spam"].astype(int)
    return df


def load_email_csv_robust(csv_path: Path, text_col_hint="text", label_col_hint="spam") -> pd.DataFrame:
    """
    Ultra-robust loader for emails.csv where:
      - header is quoted like "text,spam,,,,,,,,"
      - each data line looks like: "Subject: ....",1,,,,,,,,,,,,,
    We parse line-by-line, take the LAST comma-separated token that looks like a label (0/1/true/false),
    and treat everything before that as the email text. Anything after the label is ignored.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first = True
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Skip header-ish first line if it looks like it
            if first:
                first = False
                hdr = line.strip().strip('"').lower()
                if "text" in hdr and "spam" in hdr:
                    continue  # treat as header, skip

            # Find the LAST comma and try to interpret the token after it as the label
            last_comma = line.rfind(",")
            if last_comma == -1:
                continue  # malformed, skip

            left = line[:last_comma].strip()
            right = line[last_comma+1:].strip().strip('"').lower()

            # Normalise label token (allow trailing commas/junk)
            # e.g., "1,,,,," -> "1", "TRUE,,,," -> "true"
            if right.startswith("true"):
                label = 1
            elif right.startswith("false"):
                label = 0
            else:
                # keep only first run of digits
                num = ""
                for ch in right:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num not in {"0", "1"}:
                    # Sometimes the label is before some extra commas; try a safer split:
                    parts = [p for p in line.split(",") if p != ""]
                    if not parts:
                        continue
                    # Walk from the end to find 0/1/true/false
                    label = None
                    for tok in reversed(parts):
                        t = tok.strip().strip('"').lower()
                        if t in {"0", "1", "true", "false"}:
                            label = 1 if t in {"1", "true"} else 0
                            # rebuild text from everything before that token index
                            idx = len(parts) - 1 - list(reversed(parts)).index(tok)
                            left = ",".join(parts[:idx]).strip()
                            break
                    if label is None:
                        continue
                else:
                    label = int(num)

            # Clean quotes around the left/text
            text = left.strip().strip('"').strip()
            # Remove leading Subject:
            text = re.sub(r'^\s*Subject:\s*', "", text, flags=re.IGNORECASE)
            # Squash whitespace
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                continue

            rows.append((text, label))

    if not rows:
        raise ValueError("Could not parse any (text,label) rows from the file. Check CSV formatting.")

    df = pd.DataFrame(rows, columns=["text", "spam"])
    # Safety: drop obvious empties and ensure ints
    df = df.dropna(subset=["text", "spam"])
    df["text"] = df["text"].astype(str)
    df["spam"] = pd.to_numeric(df["spam"], errors="coerce").fillna(0).astype(int)

    # Optional sanity checks (uncomment for debugging)
    # print(df.head(5).to_string(index=False))
    # print(df["spam"].value_counts())

    return df


# ---------------------- URL task ----------------------
def suspicious_tld(u: str) -> int:
    bad = (".zip", ".top", ".xyz", ".info", ".click")
    u = (u or "").lower()
    return int(any(u.endswith(t) or (t + "/") in u for t in bad))

def has_ip(u: str) -> int:
    s = str(u)
    digits = sum(c.isdigit() for c in s)
    return int(s.count(".") >= 3 and digits >= 4)

def run_url(csv_path: Path, url_col: str, label_col: str):
    df = load_url_csv_robust(csv_path, url_col, label_col)

    feats = {}
    feats["len_url"] = df["url"].str.len()
    feats["contains_subscribe"] = df["url"].str.contains("subscribe", case=False, regex=False).astype(int)
    feats["contains_hash"] = df["url"].str.contains("#", regex=False).astype(int)
    feats["num_digits"] = df["url"].apply(lambda s: sum(c.isdigit() for c in s))
    feats["non_https"] = (~df["url"].str.startswith("https")).astype(int)
    feats["num_slashes"] = df["url"].str.count("/")
    feats["num_dots"] = df["url"].str.count(r"\.")
    feats["suspicious_tld"] = df["url"].apply(suspicious_tld)
    feats["has_ip"] = df["url"].apply(has_ip)

    X = pd.DataFrame(feats).astype(float).values
    y = df["is_spam"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipe = Pipeline([
        ("scale", MinMaxScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_SEED
        ))
    ])
    pipe.fit(Xtr, ytr)

    y_pred = pipe.predict(Xte)
    y_prob = pipe.predict_proba(Xte)[:, 1]

    print_cls_metrics("Logistic Regression (URL)", yte, y_pred, y_prob)
    save_roc_pr(yte, y_prob, "URL Model", "url_model")


# ---------------------- Email task (supervised) ----------------------
def run_email(csv_path: Path, text_col: str, label_col: str):
    df = load_email_csv_robust(csv_path, text_col_hint=text_col, label_col_hint=label_col)

    df = df[df["text"].str.contains(r"[A-Za-z]", regex=True, na=False)].copy()
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 0].copy()
    if len(df) == 0:
        raise ValueError("All rows are empty after cleaning — check the CSV formatting.")

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",    # set to None if you still see empty-vocab
        ngram_range=(1, 2),
        max_features=40000,
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32
    )
    X = tfidf.fit_transform(df["text"])
    y = df["spam"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    )
    clf.fit(Xtr, ytr)

    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)[:, 1]

    print_cls_metrics("RandomForest (Email)", yte, y_pred, y_prob)
    save_roc_pr(yte, y_prob, "Email Model", "email_model")



# ---------------------- Email task (unsupervised) ----------------------
def run_email_kmeans(csv_path: Path, text_col: str, k: int):
    """
    Quick unsupervised clustering for extra marks.
    If labels exist in the CSV (e.g., 'spam'), we show a cross-tab to discuss alignment.
    """
    # For messy CSV, just reuse the robust loader and drop labels after we get text
    df = load_email_csv_robust(csv_path, text_col_hint=text_col, label_col_hint="spam")

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        dtype=np.float32
    )
    X = tfidf.fit_transform(df["text"])

    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    clusters = km.fit_predict(X)

    print(f"\n=== KMeans (k={k}) on email TF-IDF ===")
    counts = pd.Series(clusters).value_counts().sort_index()
    print("Cluster sizes:\n", counts.to_string())

    # Optional: if label exists, compare
    if "spam" in df.columns:
        try:
            lab = df["spam"]
            print("\nContingency vs. 'spam' labels:")
            print(pd.crosstab(lab, clusters))
        except Exception as e:
            print("Could not compute crosstab:", e)
    




# ---------------------- CLI ----------------------
def main():
    p = argparse.ArgumentParser(description="Cyber ML: one-file assignment script")
    sub = p.add_subparsers(dest="task", required=True)

    p_url = sub.add_parser("url", help="URL features + Logistic Regression")
    p_url.add_argument("--csv", required=True, type=Path)
    p_url.add_argument("--url-col", default="url")
    p_url.add_argument("--label-col", default="is_spam")

    p_email = sub.add_parser("email", help="Email TF-IDF + RandomForest")
    p_email.add_argument("--csv", required=True, type=Path)
    p_email.add_argument("--text-col", default="text")
    p_email.add_argument("--label-col", default="spam")

    p_km = sub.add_parser("email-kmeans", help="Email TF-IDF + KMeans (unsupervised)")
    p_km.add_argument("--csv", required=True, type=Path)
    p_km.add_argument("--text-col", default="text")
    p_km.add_argument("--k", type=int, default=2)

    args = p.parse_args()
    np.random.seed(RANDOM_SEED)

    if args.task == "url":
        run_url(args.csv, args.url_col, args.label_col)
    elif args.task == "email":
        run_email(args.csv, args.text_col, args.label_col)
    elif args.task == "email-kmeans":
        run_email_kmeans(args.csv, args.text_col, args.k)



if __name__ == "__main__":
    main()

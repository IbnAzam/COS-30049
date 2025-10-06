"""

Tasks you can run:
  url          : URL features + Logistic Regression (classification)
  email        : Email TF-IDF + RandomForest (classification)
  email-kmeans : Email TF-IDF + KMeans (unsupervised bonus)

Examples:
  python cyberml.py url          --csv url_spam_classification.csv --url-col url --label-col is_spam
  python cyberml.py email        --csv emails.csv --text-col text --label-col spam
  python cyberml.py email-kmeans --csv emails.csv --text-col text --k 2
"""

# ---- Imports (Python + libraries) ----
import argparse                 # read command-line options like --csv path.csv
from pathlib import Path        # clean cross-platform file paths
import re                       # small text cleanups (regex)
import numpy as np              # fast arrays & math
import pandas as pd             # data tables (DataFrame)
import matplotlib.pyplot as plt # plotting charts

# scikit-learn: ML utilities
from sklearn.model_selection import train_test_split        # split data into train/test
from sklearn.preprocessing import MinMaxScaler              # scale numeric features to 0..1
from sklearn.pipeline import Pipeline                       # chain steps together
from sklearn.linear_model import LogisticRegression         # simple classifier (URL model)
from sklearn.ensemble import RandomForestClassifier         # ensemble classifier (email model)
from sklearn.metrics import (                               # evaluation metrics + plot helpers
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer # text -> numbers (email model)
from sklearn.cluster import KMeans                          # unsupervised clustering (bonus)

# Fix random seed so results are reproducible each run
RANDOM_SEED = 42


# ======================================================================
# 1) Small helpers: saving charts and printing metrics
# ======================================================================
def ensure_reports_dir():
    """Make sure the 'reports/' folder exists before saving images."""
    Path("reports").mkdir(exist_ok=True)

def save_roc_pr(y_true, y_prob, title_prefix, stem):
    """
    Draw & save ROC and Precision–Recall curves.
    - y_true: real labels (0/1)
    - y_prob: predicted probabilities for class 1
    - title_prefix: text for chart titles
    - stem: filename base like 'email_model' -> saves reports/email_model_*.png
    """
    ensure_reports_dir()

    # --- ROC curve ---
    fig1, ax1 = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax1)
    ax1.set_title(f"{title_prefix}: ROC Curve")
    fig1.tight_layout()
    fig1.savefig(f"reports/{stem}_roc.png", dpi=150)
    plt.close(fig1)  # free memory

    # --- Precision–Recall curve ---
    fig2, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax2)
    ax2.set_title(f"{title_prefix}: Precision–Recall Curve")
    fig2.tight_layout()
    fig2.savefig(f"reports/{stem}_pr.png", dpi=150)
    plt.close(fig2)

def print_cls_metrics(name, y_true, y_pred, y_prob=None):
    """
    Print common classification metrics in a friendly format.
    - y_pred: 0/1 predictions
    - y_prob: probabilities for class 1 (optional; used for ROC AUC)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")   # overall % correct
    print(f"Precision: {prec:.4f}")  # of predicted spam, how many were really spam
    print(f"Recall   : {rec:.4f}")   # of real spam, how many we caught
    print(f"F1-score : {f1:.4f}")    # balance between precision & recall

    # ROC AUC only if we have probabilities
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"ROC AUC  : {auc:.4f}")  # 1.0 perfect, 0.5 random
        except Exception:
            pass

    # Confusion matrix + per-class report (precision/recall/F1 per class)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred), "\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))


# ======================================================================
# 2) Robust loaders: turn messy CSVs into clean DataFrames
# ======================================================================
def load_url_csv_robust(csv_path, url_col="url", label_col="is_spam"):
    """
    Load a URL dataset with columns [url, is_spam].
    - Handles labels like TRUE/FALSE, spam/ham, or 0/1.
    - Drops broken or missing rows.
    Returns a DataFrame with columns: ['url', 'is_spam'] (0/1).
    """
    # Read with the 'python' engine to be forgiving with odd quoting
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    # Try to locate the desired column names (case-insensitive fallback)
    if url_col not in df.columns or label_col not in df.columns:
        cols_lower = {c.lower(): c for c in df.columns}
        url_col = cols_lower.get(url_col.lower(), url_col)
        label_col = cols_lower.get(label_col.lower(), label_col)

    # Keep only the two relevant columns and standardise their names
    df = df[[url_col, label_col]].copy()
    df.columns = ["url", "is_spam"]

    # Ensure correct types
    df["url"] = df["url"].astype(str)

    # Map string labels to 0/1
    if df["is_spam"].dtype == object:
        mapping = {
            "true": 1, "false": 0,
            "spam": 1, "ham": 0,
            "malicious": 1, "benign": 0, "safe": 0
        }
        df["is_spam"] = df["is_spam"].astype(str).str.strip().str.lower().map(mapping)

    # Force numeric 0/1, drop rows we can't parse
    df["is_spam"] = pd.to_numeric(df["is_spam"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["url", "is_spam"]).copy()
    df["is_spam"] = df["is_spam"].astype(int)
    return df


def load_email_csv_robust(csv_path, text_col_hint="text", label_col_hint="spam"):
    """
    Ultra-robust loader for emails.csv where:
      - header might be quoted like:  "text,spam,,,,,,,,"
      - rows look like:               "Subject: ....",1,,,,,,,,,
    Strategy:
      - Read the file line-by-line (manual parsing).
      - Take the LAST meaningful comma-separated token that looks like a label (0/1/true/false).
      - Everything before that is the email text.
      - Ignore extra junk after the label.
    Returns a DataFrame with columns: ['text','spam'] (0/1).
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first = True
        for raw in f:
            line = raw.strip()
            if not line:
                continue  # skip empty lines

            # Detect and skip a header line (first line) if it mentions both "text" and "spam"
            if first:
                first = False
                hdr = line.strip().strip('"').lower()
                if "text" in hdr and "spam" in hdr:
                    continue

            # Try to split off a label at the very end of the line
            last_comma = line.rfind(",")
            if last_comma == -1:
                continue  # no comma at all -> skip as malformed

            left = line[:last_comma].strip()                    # the text part (likely)
            right = line[last_comma+1:].strip().strip('"').lower()  # the tail (maybe label)

            # Quick map TRUE/FALSE first
            if right.startswith("true"):
                label = 1
            elif right.startswith("false"):
                label = 0
            else:
                # Otherwise, try to read a leading digit (e.g., "1,,,,," -> "1")
                num = ""
                for ch in right:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num in {"0", "1"}:
                    label = int(num)
                else:
                    # Fallback: walk tokens from the END to find 0/1/true/false
                    parts = [p for p in line.split(",") if p != ""]
                    if not parts:
                        continue
                    label = None
                    for tok in reversed(parts):
                        t = tok.strip().strip('"').lower()
                        if t in {"0", "1", "true", "false"}:
                            label = 1 if t in {"1", "true"} else 0
                            # Rebuild text from everything BEFORE that token
                            idx = len(parts) - 1 - list(reversed(parts)).index(tok)
                            left = ",".join(parts[:idx]).strip()
                            break
                    if label is None:
                        continue  # couldn't find a label -> skip

            # Clean the text: drop quotes, remove "Subject:", squish spaces
            text = left.strip().strip('"').strip()
            text = re.sub(r'^\s*Subject:\s*', "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue  # skip empty text

            rows.append((text, label))

    if not rows:
        raise ValueError("Could not parse any (text,label) rows from the file. Check CSV formatting.")

    # Build a clean table with two columns
    df = pd.DataFrame(rows, columns=["text", "spam"])
    df = df.dropna(subset=["text", "spam"])
    df["text"] = df["text"].astype(str)
    df["spam"] = pd.to_numeric(df["spam"], errors="coerce").fillna(0).astype(int)
    return df


# ======================================================================
# 3) URL task: features -> scale -> Logistic Regression
# ======================================================================
def suspicious_tld(u):
    """Return 1 if URL ends/contains a suspicious top-level domain (very simple rule)."""
    bad = (".zip", ".top", ".xyz", ".info", ".click")
    u = (u or "").lower()
    return int(any(u.endswith(t) or (t + "/") in u for t in bad))

def has_ip(u):
    """Return 1 if the URL looks like it contains an IP address (rough heuristic)."""
    s = str(u)
    digits = sum(c.isdigit() for c in s)
    return int(s.count(".") >= 3 and digits >= 4)

def run_url(csv_path, url_col, label_col):
    """Train/evaluate a URL spam model and save ROC/PR plots."""
    # Load & normalise the URL dataset to ['url','is_spam']
    df = load_url_csv_robust(csv_path, url_col, label_col)

    # --- Feature engineering: turn each URL string into simple numbers ---
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

    X = pd.DataFrame(feats).astype(float).values   # features -> numeric array
    y = df["is_spam"].values                       # labels -> 0/1 array

    # --- Split into train/test for fair evaluation ---
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # --- Pipeline: scale features to 0..1, then train Logistic Regression ---
    pipe = Pipeline([
        ("scale", MinMaxScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, solver="lbfgs",
            class_weight="balanced",      # handle class imbalance
            random_state=RANDOM_SEED
        ))
    ])
    pipe.fit(Xtr, ytr)

    # --- Predict on test set ---
    y_pred = pipe.predict(Xte)              # hard 0/1 decisions
    y_prob = pipe.predict_proba(Xte)[:, 1]  # probabilities for class 1

    # --- Print metrics + save charts ---
    print_cls_metrics("Logistic Regression (URL)", yte, y_pred, y_prob)
    save_roc_pr(yte, y_prob, "URL Model", "url_model")


# ======================================================================
# 4) Email task (supervised): TF-IDF -> Random Forest
# ======================================================================
def run_email(csv_path, text_col, label_col):
    """Train/evaluate an email spam model on messy emails.csv and save ROC/PR plots."""
    # Robust load -> clean ['text','spam'] table
    df = load_email_csv_robust(csv_path, text_col_hint=text_col, label_col_hint=label_col)

    # Extra safety: keep rows that still have letters after cleaning
    df = df[df["text"].str.contains(r"[A-Za-z]", regex=True, na=False)].copy()
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 0].copy()
    if len(df) == 0:
        raise ValueError("All rows are empty after cleaning — check the CSV formatting.")

    # Vectorise text -> numbers (TF-IDF)
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",     # set to None if you ever hit empty-vocabulary errors
        ngram_range=(1, 2),       # unigrams + bigrams
        max_features=40000,       # reduce if RAM is tight
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32          # smaller memory per value
    )
    X = tfidf.fit_transform(df["text"])     # sparse matrix of features
    y = df["spam"].values                   # 0/1 labels

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Random Forest classifier (robust baseline)
    clf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    )
    clf.fit(Xtr, ytr)

    # Predict on test set
    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)[:, 1]

    # Metrics + charts
    print_cls_metrics("RandomForest (Email)", yte, y_pred, y_prob)
    save_roc_pr(yte, y_prob, "Email Model", "email_model")


# ======================================================================
# 5) Email task (unsupervised bonus): TF-IDF -> KMeans
# ======================================================================
def run_email_kmeans(csv_path, text_col, k):
    """
    Unsupervised clustering (no training labels needed).
    Good for discussion/bonus marks: do clusters align with spam/ham?
    """
    # Reuse robust loader to get clean 'text' (labels optional)
    df = load_email_csv_robust(csv_path, text_col_hint=text_col, label_col_hint="spam")

    # TF-IDF vectorise the text
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

    # KMeans clustering into k groups
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    clusters = km.fit_predict(X)

    print(f"\n=== KMeans (k={k}) on email TF-IDF ===")
    counts = pd.Series(clusters).value_counts().sort_index()
    print("Cluster sizes:\n", counts.to_string())

    # If a 'spam' label column exists, show a quick cross-tab for discussion
    if "spam" in df.columns:
        try:
            lab = df["spam"]
            print("\nContingency vs. 'spam' labels:")
            print(pd.crosstab(lab, clusters))
        except Exception as e:
            print("Could not compute crosstab:", e)


# ======================================================================
# 6) CLI entry point: choose a task and run it
# ======================================================================
def main():
    """
    Command-line interface.
    Pick a task (url / email / email-kmeans) and pass the dataset path.
    """
    p = argparse.ArgumentParser(description="Cyber ML: one-file assignment script")
    sub = p.add_subparsers(dest="task", required=True)

    # Subcommand: URL classification
    p_url = sub.add_parser("url", help="URL features + Logistic Regression")
    p_url.add_argument("--csv", required=True, type=Path)      # dataset file path
    p_url.add_argument("--url-col", default="url")             # column name for URLs
    p_url.add_argument("--label-col", default="is_spam")       # column name for labels

    # Subcommand: Email classification
    p_email = sub.add_parser("email", help="Email TF-IDF + RandomForest")
    p_email.add_argument("--csv", required=True, type=Path)
    p_email.add_argument("--text-col", default="text")         # email text column
    p_email.add_argument("--label-col", default="spam")        # label column

    # Subcommand: Email KMeans (unsupervised)
    p_km = sub.add_parser("email-kmeans", help="Email TF-IDF + KMeans (unsupervised)")
    p_km.add_argument("--csv", required=True, type=Path)
    p_km.add_argument("--text-col", default="text")
    p_km.add_argument("--k", type=int, default=2)              # number of clusters

    args = p.parse_args()
    np.random.seed(RANDOM_SEED)  # reproducible splits/models

    # Route to the chosen task
    if args.task == "url":
        run_url(args.csv, args.url_col, args.label_col)
    elif args.task == "email":
        run_email(args.csv, args.text_col, args.label_col)
    elif args.task == "email-kmeans":
        run_email_kmeans(args.csv, args.text_col, args.k)


# Standard Python entry point: only run main() when executing this file directly.
if __name__ == "__main__":
    main()

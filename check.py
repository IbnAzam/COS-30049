# check.py — Spam detection with Logistic Regression only

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# -------------------------
# 1) Load data
# -------------------------
df = pd.read_csv("url_spam_classification.csv")

# If labels are strings like "spam"/"ham", map them to 1/0.
if df["is_spam"].dtype == object:
    df["is_spam"] = df["is_spam"].str.lower().map({"spam": 1, "ham": 0}).astype(int)

# -------------------------
# 2) Minimal, fast features (all numeric)
# -------------------------
df["len_url"] = df["url"].str.len()
df["contains_subscribe"] = df["url"].str.contains("subscribe", case=False, regex=False).astype(int)
df["contains_hash"] = df["url"].str.contains("#", regex=False).astype(int)
df["num_digits"] = df["url"].apply(lambda s: sum(c.isdigit() for c in s))
# Name says non_https ⇒ 1 if NOT https, else 0
df["non_https"] = (~df["url"].str.startswith("https")).astype(int)
df["num_words"] = df["url"].apply(lambda s: len(s.split("/")))

target = "is_spam"
features = ["len_url", "contains_subscribe", "contains_hash", "num_digits", "non_https", "num_words"]

X = df[features].astype(float)
y = df[target].astype(int)

# -------------------------
# 3) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# -------------------------
# 4) Logistic Regression pipeline
# -------------------------
pipe = Pipeline([
    ("scale", MinMaxScaler()),
    ("lr", LogisticRegression(max_iter=1000, solver="lbfgs"))
])
pipe.fit(X_train, y_train)

# -------------------------
# 5) Evaluation
# -------------------------
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("=== Logistic Regression (Spam vs Non-Spam) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# -------------------------
# 6) Example: predict on a few URLs (optional)
# -------------------------
sample_urls = [
    "https://example.com/account/subscribe?offer=free",
    "http://192.168.0.1/login#reset",
    "https://university.edu/resources/notes/week-1.pdf",
]
def featurize(url: str) -> dict:
    return {
        "len_url": len(url),
        "contains_subscribe": int("subscribe" in url.lower()),
        "contains_hash": int("#" in url),
        "num_digits": sum(c.isdigit() for c in url),
        "non_https": int(not url.startswith("https")),
        "num_words": len(url.split("/")),
    }

X_new = pd.DataFrame([featurize(u) for u in sample_urls])[features].astype(float)
preds = pipe.predict(X_new)
probs = pipe.predict_proba(X_new)[:, 1]

print("\n=== Sample predictions ===")
for u, p, pr in zip(sample_urls, preds, probs):
    label = "SPAM" if p == 1 else "NOT SPAM"
    print(f"{label:9s}  ({pr:.3f} prob)  {u}")


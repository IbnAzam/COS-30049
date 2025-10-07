import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Regression Technique



# --- Load data ---
df = pd.read_csv("final.csv")
assert {"content", "label"}.issubset(df.columns), "final.csv must have columns: content, label"
df = df.dropna(subset=["content", "label"]).copy()
df["label"] = df["label"].astype(int)

X = df["content"]
y = df["label"]

# --- Split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- TF-IDF ---
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1,2),
    lowercase=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# Some sklearn builds accept sparse CSR for forests; if you hit an error or it’s slow,
# uncomment the next two lines to densify (watch RAM if dataset is large).
# X_train_tfidf = X_train_tfidf.toarray()
# X_test_tfidf  = X_test_tfidf.toarray()

# --- Random Forest ---
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,            # let trees grow; you can cap (e.g., 20) if overfitting
    min_samples_leaf=1,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train_tfidf, y_train)

# --- Evaluate ---
y_pred = rf.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))

# Optional: probabilities for threshold tuning (e.g., maximize spam recall)
# y_proba = rf.predict_proba(X_test_tfidf)[:, 1]

# Optional: top "important" tokens (often noisy for text, but can be interesting)
# import numpy as np
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1][:20]
# feats = vectorizer.get_feature_names_out()
# top = [(feats[i], float(importances[i])) for i in indices]
# print("Top features by importance:", top)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Classification Model



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

# --- Logistic Regression ---
model_lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # helps if spam/ham are imbalanced
    solver="liblinear"         # good for small/medium datasets
)
model_lr.fit(X_train_tfidf, y_train)

# --- Evaluate ---
y_pred = model_lr.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))

# Optional: predicted probabilities (useful for threshold tuning)
# y_proba = model_lr.predict_proba(X_test_tfidf)[:, 1]

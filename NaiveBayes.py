import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_auc_score


# Classification Technique


# --- Load data ---
df = pd.read_csv("final.csv")

# Basic sanity checks / cleanup
assert {"content", "label"}.issubset(df.columns), "final.csv must have columns: content, label"
df = df.dropna(subset=["content", "label"]).copy()
# Ensure labels are 0/1 integers
df["label"] = df["label"].astype(int)

X = df["content"]
y = df["label"]

# --- Train / test split (stratify to preserve spam/ham ratio) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Vectorize text with TF-IDF ---
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1,2),        # unigrams + bigrams often help for spam
    lowercase=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# --- Train Multinomial Naive Bayes ---
# alpha is smoothing; 0.1–1.0 are common good values
model_nb = MultinomialNB(alpha=0.3)
model_nb.fit(X_train_tfidf, y_train)

# --- Evaluate ---
y_pred = model_nb.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))

# Optionally, predicted probabilities (useful if you want to tune a threshold)
# y_proba = model_nb.predict_proba(X_test_tfidf)[:, 1]





# --- Confusion Matrix (Naive Bayes) ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix – Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Ham (0)", "Spam (1)"])
plt.yticks([0,1], ["Ham (0)", "Spam (1)"])

# Annotate counts
for (i, j), v in [((i,j), cm[i,j]) for i in range(cm.shape[0]) for j in range(cm.shape[1])]:
    plt.text(j, i, str(v), ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig("confusion_matrix_nb.png", dpi=150)
plt.close()

# --- ROC Curve (Naive Bayes) ---
y_prob = model_nb.predict_proba(X_test_tfidf)[:, 1]   # spam=1 probability
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve – Naive Bayes")
plt.savefig("roc_curve_nb.png", dpi=150)
plt.close()

# AUC (Area Under ROC)
auc = roc_auc_score(y_test, y_prob)
print(f"\nAUC (NB): {auc:.4f}")

print("\nVisualisations saved as:")
print(" confusion_matrix_nb.png")
print(" roc_curve_nb.png")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay

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




cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks([0,1], ["Ham (0)", "Spam (1)"])
plt.yticks([0,1], ["Ham (0)", "Spam (1)"])
for (i, j), v in [((i,j), cm[i,j]) for i in range(cm.shape[0]) for j in range(cm.shape[1])]:
    plt.text(j, i, str(v), ha="center", va="center")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png", dpi=150)
plt.close()


y_prob = model_lr.predict_proba(X_test_tfidf)[:, 1]
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve – Logistic Regression")
plt.savefig("roc_curve_lr.png", dpi=150)
plt.close()

print("\n Visualisations saved as:")
print(" confusion_matrix_lr.png")
print(" roc_curve_lr.png")

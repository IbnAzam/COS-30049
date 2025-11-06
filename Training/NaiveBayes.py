# NaiveBayes.py (refactor)
import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

OUT_DIR = "NaiveBayesModel"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("final.csv")
assert {"content", "label"}.issubset(df.columns)
df = df.dropna(subset=["content", "label"]).copy()
df["label"] = df["label"].astype(int)

X = df["content"].astype(str)
y = df["label"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        #stop_words="english",
        ngram_range=(1,2),
        lowercase=True,
        sublinear_tf=True,          # often helps
        strip_accents="unicode",
        max_features=20000          # give vocab some headroom
    )),
    ("nb", MultinomialNB(alpha=0.3)),
])

pipe.fit(X_tr, y_tr)

y_pred = pipe.predict(X_te)
print(f"Accuracy: {accuracy_score(y_te, y_pred):.4f}\n")
print(classification_report(y_te, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))

joblib.dump(pipe, f"{OUT_DIR}/spam_nb_pipeline.pkl")
print(f"\n💾 Saved {OUT_DIR}/spam_nb_pipeline.pkl")

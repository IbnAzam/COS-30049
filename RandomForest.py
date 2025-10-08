

import numpy as np
import pandas as pd



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import confusion_matrix, RocCurveDisplay



# --- Visualise model performance ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay
# --- Load data ---
df = pd.read_csv("final.csv")

# Basic cleanup (avoid NaNs)
df["content"] = df["content"].astype(str).str.replace("\r\n", " ", regex=False).str.strip()

# Labels (handle "spam"/"ham" or 0/1)
y_raw = df["label"]
if y_raw.dtype == object:
    y = y_raw.astype(str).str.lower().map({"spam": 1, "ham": 0}).astype(int).values
else:
    y = y_raw.astype(int).values

# --- Vectorise (keep it small for RF speed) ---
# Reduce feature space so RF doesn't blow up memory/time
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=5000,   
    min_df=3             # drop super-rare words
)
X = vectorizer.fit_transform(df["content"])

print("Matrix shape:", X.shape)  # (rows, vocab)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model (RandomForest) ---
clf = RandomForestClassifier(
    n_estimators=100,     
    max_depth=200,         # cap depth for speed; set None to grow fully (slower)
    max_features="sqrt",  # standard RF setting
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

# --- Classify a sample email (e.g., row 10) ---
idx = 10
if 0 <= idx < len(df):
    email_text = str(df["content"].iloc[idx])
    X_email = vectorizer.transform([email_text])
    pred = clf.predict(X_email)[0]
    print(f"\nRow {idx} -> Predicted: {pred} | Actual: {df['label'].iloc[idx]}")








# --- Confusion Matrix (Random Forest) ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix – Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Ham (0)", "Spam (1)"])
plt.yticks([0,1], ["Ham (0)", "Spam (1)"])

# Annotate cell counts
for (i, j), v in [((i,j), cm[i,j]) for i in range(cm.shape[0]) for j in range(cm.shape[1])]:
    plt.text(j, i, str(v), ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig("confusion_matrix_rf.png", dpi=150)
plt.close()

# --- ROC Curve (Random Forest) ---
y_prob = clf.predict_proba(X_test)[:, 1]   
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve – Random Forest")
plt.savefig("roc_curve_rf.png", dpi=150)
plt.close()

print("\nVisualisations saved as:")
print(" confusion_matrix_rf.png")
print(" roc_curve_rf.png")

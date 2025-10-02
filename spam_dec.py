import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Ensure stopwords are available
nltk.download('stopwords')

# --- Load data ---
df = pd.read_csv('emails.csv')

# Basic cleanup (avoid NaNs breaking lower()/translate())
df['text'] = df['text'].astype(str).str.replace('\r\n', ' ', regex=False)

# --- Preprocessing setup ---
stemmer = PorterStemmer()                         # instantiate!
stopwords_set = set(stopwords.words('english'))   # fix spelling
punct_table = str.maketrans('', '', string.punctuation)

def preprocess(s: str):
    s = s.lower().translate(punct_table)
    tokens = [stemmer.stem(w) for w in s.split() if w not in stopwords_set]
    return ' '.join(tokens)

# Build corpus
corpus = [preprocess(t) for t in df['text']]

# --- Vectorise ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)  # keep name consistent
y = df['spam'].values            # assumes this column exists (0/1)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model ---
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

print("Test accuracy:", clf.score(X_test, y_test))

# --- Classify a sample email (e.g., row 10) ---
email_to_classify = str(df['text'].iloc[10])
email_processed = preprocess(email_to_classify)
X_email = vectorizer.transform([email_processed])

pred = clf.predict(X_email)[0]
print("Predicted:", pred, " | Actual:", df['spam'].iloc[10])

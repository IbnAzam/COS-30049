
import joblib, pandas as pd
from sklearn.pipeline import Pipeline


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer




urls = pd.read_csv("urls_only.csv")
pipe_url = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])
pipe_url.fit(urls["content"], urls["label"])
joblib.dump(pipe_url, "Models/url_model.pkl")

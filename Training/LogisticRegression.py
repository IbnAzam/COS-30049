from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib, pandas as pd



# Email model
emails = pd.read_csv("emails_only.csv")
pipe_email = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        sublinear_tf=True,
        max_features=20000
    )),
    ("lr", LogisticRegression(max_iter=2000))
])
pipe_email.fit(emails["content"], emails["label"])
joblib.dump(pipe_email, "Models/email_model.pkl")



# URL model  (note: analyzer="char_wb")
urls = pd.read_csv("urls_only.csv")
pipe_url = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,5),
        min_df=2,
        max_features=200000
    )),
    ("lr", LogisticRegression(max_iter=2000))
])
pipe_url.fit(urls["content"], urls["label"])
joblib.dump(pipe_url, "Models/url_model.pkl")

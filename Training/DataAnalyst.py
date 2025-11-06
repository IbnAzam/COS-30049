import pandas as pd
import re, string
from collections import Counter

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Plots
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- Setup NLTK (do once) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# --- 1) Load ---
combined = pd.read_csv("final.csv")
assert {"content", "label"}.issubset(combined.columns)

# --- 2) Clean text ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [t for t in text.split() if t and t not in stop_words]
    # optional: lemmatize (comment out if slow)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

combined["clean_text"] = combined["content"].map(clean_text)

print("✅ Data cleaned")

# ---------------------------
# A) WORD CLOUDS (saved)
# ---------------------------
spam_texts = " ".join(combined.loc[combined["label"] == 1, "clean_text"])
ham_texts  = " ".join(combined.loc[combined["label"] == 0, "clean_text"])

wc = WordCloud(max_words=200, background_color="white")

plt.figure(figsize=(6,5))
plt.imshow(wc.generate(spam_texts))
plt.axis("off")
plt.title("Spam Word Cloud")
plt.tight_layout()
plt.savefig("spam_wordcloud.png", dpi=150)
plt.close()

plt.figure(figsize=(6,5))
plt.imshow(wc.generate(ham_texts))
plt.axis("off")
plt.title("Ham Word Cloud")
plt.tight_layout()
plt.savefig("ham_wordcloud.png", dpi=150)
plt.close()

# ---------------------------
# B) TOP TOKENS BAR CHARTS
# ---------------------------
def top_tokens(text_series, k=20):
    cnt = Counter()
    for line in text_series:
        cnt.update(line.split())
    return cnt.most_common(k)

top_spam = top_tokens(combined.loc[combined["label"] == 1, "clean_text"])
top_ham  = top_tokens(combined.loc[combined["label"] == 0, "clean_text"])

def save_top_tokens_bar(pairs, title, filename):
    if not pairs:
        return
    tokens, freqs = zip(*pairs)
    plt.figure(figsize=(10,5))
    # horizontal bars for readability
    y = range(len(tokens))
    plt.barh(y, freqs)
    plt.yticks(y, tokens)
    plt.xlabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

save_top_tokens_bar(top_spam, "Top Tokens – Spam", "top_tokens_spam.png")
save_top_tokens_bar(top_ham,  "Top Tokens – Ham",  "top_tokens_ham.png")

# ---------------------------
# C) MESSAGE LENGTH HISTOGRAM
# ---------------------------
combined["msg_len"] = combined["clean_text"].str.split().map(len)

plt.figure(figsize=(8,5))
# plot both distributions on the same axes
combined.loc[combined["label"] == 0, "msg_len"].plot(kind="hist", bins=50, alpha=0.6, label="Ham (0)")
combined.loc[combined["label"] == 1, "msg_len"].plot(kind="hist", bins=50, alpha=0.6, label="Spam (1)")
plt.legend()
plt.xlabel("Token Count per Message")
plt.title("Message Length Distribution")
plt.tight_layout()
plt.savefig("msg_length_hist.png", dpi=150)
plt.close()

print("\nVisualisations saved as:")
print(" spam_wordcloud.png")
print(" ham_wordcloud.png")
print(" top_tokens_spam.png")
print(" top_tokens_ham.png")
print(" msg_length_hist.png")

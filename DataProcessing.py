import pandas as pd

# Load both CSVs
df1 = pd.read_csv("emails.csv")                  # columns: text, spam
df2 = pd.read_csv("url_spam_classification.csv") # columns: url, is_spam

# 1️⃣ Rename columns to match the same schema
df1 = df1.rename(columns={"text": "content", "spam": "label"})
df2 = df2.rename(columns={"url": "content", "is_spam": "label"})

# 2️⃣ Convert df2's 'label' (True/False) to 1/0 to match df1
df2["label"] = df2["label"].astype(int)

# 3️⃣ Combine both into one DataFrame
combined = pd.concat([df1, df2], ignore_index=True)

# 4️⃣ Save result
combined.to_csv("final.csv", index=False)

# Remove any row that has at least one null value
combined = combined.dropna()

print("✅ Combined dataset saved as final.csv")


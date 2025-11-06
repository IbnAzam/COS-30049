import pandas as pd

combined = pd.read_csv("final.csv")

# Identify rows that are URLs vs. regular text
is_url = combined["content"].str.startswith("http", na=False)

# Split into two datasets
emails_df = combined[~is_url]
urls_df = combined[is_url]

emails_df.to_csv("emails_only.csv", index=False)
urls_df.to_csv("urls_only.csv", index=False)

print(f"Emails: {len(emails_df)} rows")
print(f"URLs  : {len(urls_df)} rows")

import pandas as pd


# Load both CSVs
df1 = pd.read_csv("file1.csv")
df2 = pd.read_csv("file2.csv")

# Merge based on a common column, e.g. 'id'
merged = pd.merge(df1, df2, on="id", how="inner")

# Or concatenate if they have the same columns
combined = pd.concat([df1, df2], ignore_index=True)

# Save result
merged.to_csv("final.csv", index=False)
# Spam Email Dataset

## Overview
This dataset contains emails labeled as either **ham (legitimate)** or **spam (junk email)**.

- **Columns**  
  - `text`: The content of the email  
  - `spam`: Label (0 = ham, 1 = spam)  

## Labels
- **0 → Ham**: Normal, non-spam emails  
- **1 → Spam**: Junk or unwanted emails  

## Feature Extraction Ideas
Students may extract features such as:
- Email length (characters, words)  
- Frequency of special characters (`!`, `$`, etc.)  
- Presence of keywords (e.g., *free, win, offer*)  
- Word frequency or TF-IDF values  

## Visualization Ideas
- Distribution of email lengths (spam vs. ham)  
- Word clouds for spam and ham  
- Histogram of keyword frequencies  

## How to use
- Run individual Machine Learning Model scripts on their own (generates png of visuals), there are three
  1. LinearRegression.py
  2. NaiveBayes.py
  3. RandomForest.py

- DataProcessing.py is merging the two separate CSVs and makiing common columns to align them into one coherent set
  - Creates final.csv

- DataAnalyst.py is data visualisation of the final.csv

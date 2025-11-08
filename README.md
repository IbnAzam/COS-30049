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
you will need two terminals running at once for this to work

**first terminal:**
`cd BackEnd`

then
`.venv\Scripts\activate`

for the python libraries 
(install them via the requirements.txt - `pip install -r requirements.txt` )
then
`uvicorn app.main:app --reload --port 8000`



**second terminal:**
`cd FrontEnd`

then
`npm run dev`

Go to http://localhost:5173/
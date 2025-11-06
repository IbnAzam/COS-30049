import joblib

email_model = joblib.load("Models/email_model.pkl")
url_model   = joblib.load("Models/url_model.pkl")

def classify_input(text):
    # Pick which model to use
    model = url_model if text.startswith("http") else email_model

    # Predict class and probability
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0][1]  # probability that it's spam (class 1)

    label = "Spam" if pred == 1 else "Ham"
    return f"{label} ({prob*100:.2f}% confidence)"


# Example: path to your text file (adjust to your actual file location)
path = r"C:\Users\David\Desktop\input.txt"

# Read the whole file as a string
with open(path, "r", encoding="utf-8") as file:
    testEmail = file.read()

print(testEmail)  # optional: check what got read

# Example tests
print(classify_input(testEmail))
print(classify_input("https://www.youtube.com/shorts/LSt1RIhecHU"))

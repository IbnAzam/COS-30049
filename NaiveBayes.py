#this multinomial naive bayes algorithm program is largely based on the guide on geeksforgeeks
#written by: susmit_sekhar_bhakta
#https://www.geeksforgeeks.org/machine-learning/multinomial-naive-bayes/ 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("final.csv")
df = df.dropna(subset=["content", "label"]).copy()

#split data
X = df['content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#convert text data into numerical vectors
vector = CountVectorizer()
v_X_train = vector.fit_transform(X_train) #this learns "trains" the vocabulary
v_X_test = vector.transform(X_test) #convert test data to same feature space 

algorithm = MultinomialNB(alpha=0.3) #train model using vectorised data 
algorithm.fit(v_X_train, y_train)
y_pred = algorithm.predict(v_X_test) #generate predictions

print(f"Accuracy: {accuracy_score(y_test,y_pred):.4}\n") #print results
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))

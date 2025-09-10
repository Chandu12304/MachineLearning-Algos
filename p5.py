import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score

data=pd.read_csv('train.csv')

df=pd.DataFrame(data,columns=["label","text"])
df["label"]=df["label"].map({'ham':0,'spam':1})

print(df)

x_train,x_test,y_train,y_test=train_test_split(df["text"],df["label"],test_size=0.3,random_state=42)

print(x_train)
print(x_test)

vectorizer=CountVectorizer(stop_words='english')
x_train_vec=vectorizer.fit_transform(x_train)
x_test_vec=vectorizer.transform(x_test)

model=MultinomialNB()
model.fit(x_train_vec,y_train)

y_pred=model.predict(x_test_vec)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,zero_division=1)
recall=recall_score(y_test,y_pred,zero_division=1)

print("===Evalution===")
print(f"Accuracy: {accuracy:.2f}")
print(f"precision: {precision:.2f}")
print(f"recall: {recall:.2f}")


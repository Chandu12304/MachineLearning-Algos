import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = [
    ['ham', "Hey, are we still on for dinner tonight?"],
    ['spam', "WINNER! You have won a free cruise. Call now!"],
    ['ham', "I'll call you back in 10 minutes."],
    ['spam', "URGENT! Your account has been suspended. Click to verify."],
    ['ham', "Don't forget about the meeting at 3 PM."],
    ['spam', "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May"],
    ['ham', "Lunch tomorrow?"],
    ['spam', "Claim your free ringtone now by texting WIN to 80085"],
    ['ham', "Can you send me the report before noon?"],
    ['spam', "Congratulations! You've been selected for a $1000 Walmart gift card."]
]

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


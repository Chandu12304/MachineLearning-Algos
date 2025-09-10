#KNearestNeighbors
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = [
    [170, 70, "male"],
    [165, 55, "female"],
    [180, 80, "male"],
    [175, 75, "male"],
    [160, 50, "female"],
    [155, 45, "female"],
    [185, 90, "male"],
    [170, 65, "female"]
]

df=pd.DataFrame(data,columns=["height","weight","gender"])
x=df[["height","weight"]]
y=df["gender"]

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)

h=int(input("height: "))
w=int(input("weight: "))

new_data=pd.DataFrame([[h,w]],columns=["height","weight"])

pred=knn.predict(new_data)[0]
print(pred)

colors={"male":"blue","female":"red"}
plt.scatter(x["height"], x["weight"], c=y.map(colors))
plt.scatter(h, w, c="green", s=100, marker='*') 
plt.grid()
plt.show()

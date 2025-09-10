import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Dataset
data = [
    [170, 70, "Male"],
    [165, 55, "Female"],
    [180, 80, "Male"],
    [175, 75, "Male"],
    [160, 50, "Female"],
    [155, 45, "Female"],
    [185, 90, "Male"],
    [170, 65, "Female"]
]
df = pd.DataFrame(data, columns=["Height", "Weight", "Gender"])

# Train KNN on full dataset
X = df[["Height", "Weight"]]
y = df["Gender"]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Take user input
h = int(input("Enter Height: "))
w = int(input("Enter Weight: "))

# Predict
print("Predicted Gender:", knn.predict([[h, w]])[0])

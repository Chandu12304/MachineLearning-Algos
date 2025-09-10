import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Represent the data as a list of lists
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

# Step 2: Convert to DataFrame
df = pd.DataFrame(data, columns=["Height", "Weight", "Gender"])

print("Training Data:")
print(df)

# Step 3: Features and Labels
X = df[["Height", "Weight"]]
y = df["Gender"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 5: KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\nTest Data:")
print(X_test)
print("\nActual Labels:")
print(y_test.values)
print("\nPredicted Labels:")
print(y_pred)

# Step 6: Compare Predictions
correct = X_test[y_test == y_pred]
wrong = X_test[y_test != y_pred]

print("\nCorrect Predictions:")
print(correct)
print("\nWrong Predictions:")
print(wrong)

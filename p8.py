# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Define a simple dataset
# Features: [Height (cm), Weight (kg)]
# Labels: 'Male' or 'Female'
data = pd.DataFrame({
    'Height': [170, 165, 180, 175, 160, 155, 185, 170],
    'Weight': [70, 55, 80, 75, 50, 45, 90, 65],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female']
})

print("Training Data:")
print(data)

# Step 2: Split features and labels
X = data[['Height', 'Weight']]  # Features
y = data['Gender']              # Labels

# Step 3: Split into training and test sets (for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4: Initialize k-NN classifier (k=3)
knn = KNeighborsClassifier(n_neighbors=3)

# Step 5: Train the classifier
knn.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = knn.predict(X_test)

# Step 7: Print correct and wrong predictions
print("\nTest Data:")
print(X_test)
print("\nActual Labels:")
print(y_test.values)
print("\nPredicted Labels:")
print(y_pred)

# Step 8: Identify correct and wrong predictions
correct = X_test[y_test == y_pred]
wrong = X_test[y_test != y_pred]

print("\nCorrect Predictions:")
print(correct)

print("\nWrong Predictions:")
print(wrong)

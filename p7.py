import pandas as pd
from sklearn.cluster import KMeans

# Step 1: Define the training data as a list of lists
data = [
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["X", "Y"])

print("Data:")
print(df)

# Step 2: Initialize K-Means (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0)

# Step 3: Train the model
kmeans.fit(df)

# Step 4: Get cluster labels
labels = kmeans.labels_
print("\nCluster labels for each data point:")
print(labels)

# Step 5: Get cluster centers
centers = kmeans.cluster_centers_
print("\nCluster centers:")
print(centers)

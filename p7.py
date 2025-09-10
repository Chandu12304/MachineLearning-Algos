import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Generate random training data
np.random.seed(0)
x = np.linspace(0, 10, 100)        # 100 values from 0 to 10
y = 2 * x + np.random.randn(100)   # Linear relation with noise

# Convert to DataFrame
df = pd.DataFrame({"X": x, "Y": y})
print("Data:")
print(df.head())  # print first 5 rows only

# Step 2: Initialize K-Means (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0)

# Step 3: Train the model
kmeans.fit(df)

# Step 4: Get cluster labels
labels = kmeans.labels_
print("\nCluster labels for first 10 data points:")
print(labels[:10])

# Step 5: Get cluster centers
centers = kmeans.cluster_centers_
print("\nCluster centers:")
print(centers)

# Step 6: Graphical Representation
plt.scatter(df.X, df.Y, c=labels)      # points
plt.scatter(*centers.T, c="red", s=200, marker="X")  # centers
plt.grid()
plt.show()

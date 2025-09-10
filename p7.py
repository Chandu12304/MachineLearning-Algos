import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=[[2.74, 3.58], [3.92, 0.93], [2.92, 2.19], [2.71, 4.01],
 [1.36, 1.27], [4.37, 2.9], [0.71, 4.46], [4.19, 1.18],
 [2.98, 4.49], [2.23, 3.88]]

df=pd.DataFrame(data,columns=["X","Y"])
print(df)

kmeans=KMeans(n_clusters=2,random_state=0)
kmeans.fit(df)

labels=kmeans.labels_
print(labels)

centers=kmeans.cluster_centers_
print(centers)

plt.scatter(df.X,df.Y,c=labels)
plt.scatter(*centers.T,c="red",s=100,marker="X")
plt.grid()
plt.show()

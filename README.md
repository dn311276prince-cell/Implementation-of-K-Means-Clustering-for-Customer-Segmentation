# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. **Initialize the number of clusters (K)**
   Decide the number of customer segments (K) and randomly select K data points as initial cluster centroids.

2. **Assign customers to the nearest centroid**
   Calculate the distance (usually Euclidean distance) between each customer and all centroids, then assign each customer to the closest cluster.

3. **Update cluster centroids**
   Recalculate the centroid of each cluster by taking the mean of all customers assigned to that cluster.

4. **Repeat until convergence**
   Repeat steps 2 and 3 until the centroids no longer change or customer assignments remain stable, forming final customer segments.



## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: dilip kumar R
RegisterNumber:25017135
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Gender': ['Male','Female','Female','Male','Female','Male','Male','Female','Female','Male'],
    'Age': [19,21,20,23,31,22,35,30,25,28],
    'Annual Income (k$)': [15,16,17,18,19,20,21,22,23,24],
    'Spending Score (1-100)': [39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Select features for clustering
# ------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ------------------------------
# Step 3: Apply K-Means (choose clusters, e.g., 3)
# ------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)  # Automatically fits and assigns clusters

# ------------------------------
# Step 4: Visualize clusters
# ------------------------------
plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(X[df['Cluster']==i]['Annual Income (k$)'],
                X[df['Cluster']==i]['Spending Score (1-100)'],
                label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='yellow', label='Centroids', marker='X')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# ------------------------------
# Step 5: Show dataset with clusters
# ------------------------------
print(df)
  
*/
```

## Output:
<Figure size 800x600 with 1 Axes><img width="686" height="547" alt="image" src="https://github.com/user-attachments/assets/733f8698-fe9e-4656-ad6f-ce6af195ce8d" />
CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)  \
0           1    Male   19                  15                      39   
1           2  Female   21                  16                      81   
2           3  Female   20                  17                       6   
3           4    Male   23                  18                      77   
4           5  Female   31                  19                      40   
5           6    Male   22                  20                      76   
6           7    Male   35                  21                       6   
7           8  Female   30                  22                      94   
8           9  Female   25                  23                       3   
9          10    Male   28                  24                      72   

   Cluster  
0        2  
1        0  
2        1  
3        0  
4        2  
5        0  
6        1  
7        0  
8        1  
9        0


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

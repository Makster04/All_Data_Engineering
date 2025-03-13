### **K-Means Clustering Overview**
K-Means is a popular **unsupervised machine learning algorithm** used for **clustering data into K groups**. It assigns data points to clusters based on their similarity, measured using **Euclidean distance**. The algorithm is widely used in customer segmentation, image recognition, and anomaly detection.

---

## **Clustering Methods: A Comparison**
There are multiple clustering techniques, each with unique strengths and weaknesses:

| **Method**            | **Description**  | **Pros**  | **Cons** |
|----------------------|----------------|----------|----------|
| **K-Means** | Partitions data into **K clusters** based on Euclidean distance | Fast, scalable, simple | Sensitive to outliers and initial centroid placement |
| **Hierarchical Clustering** | Creates a **tree-like structure (dendrogram)** of clusters | No need to specify K | Computationally expensive for large datasets |
| **DBSCAN** | Groups data based on **density** rather than distance | Detects arbitrarily shaped clusters, handles noise well | Struggles with varying densities and high-dimensional data |
| **Gaussian Mixture Model (GMM)** | Uses **probabilistic clustering** based on Gaussian distributions | Works well with overlapping clusters | Requires choosing K, computationally intensive |

---

## **Steps in the K-Means Algorithm**
1. **Select K:** Choose the number of clusters.
2. **Initialize Centroids:** Randomly select **K points** as the initial cluster centers.
3. **Assign Data Points:** Each data point is assigned to the **nearest centroid**.
4. **Update Centroids:** Compute the new centroids as the **mean of all assigned points**.
5. **Repeat:** Iterate until centroids no longer change (convergence).

---

## **Implementing K-Means in Scikit-Learn**
Below is an example of applying K-Means clustering on NBA player statistics.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate synthetic NBA dataset
np.random.seed(42)
num_players = 100
points = np.random.randint(5, 30, num_players)
assists = np.random.randint(1, 10, num_players)
rebounds = np.random.randint(2, 15, num_players)

# Create DataFrame
nba_df = pd.DataFrame({'Points': points, 'Assists': assists, 'Rebounds': rebounds})

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(nba_df)

# Apply K-means with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
nba_df['Cluster'] = kmeans.fit_predict(features_scaled)

# Display results
print(nba_df.head())
```
👉 **This segments NBA players into three clusters based on performance.**

---

## **Evaluating Clusters**
### **1. Inertia (Within-Cluster Sum of Squares, WCSS)**
- Measures **compactness** of clusters.
- **Lower WCSS** indicates better clustering.

```python
wcss = kmeans.inertia_
print("WCSS:", wcss)
```

---

### **2. Silhouette Score**
- Measures **how similar** a data point is to its assigned cluster versus other clusters.
- **Ranges from -1 to 1** (higher values indicate better-defined clusters).

```python
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(features_scaled, nba_df['Cluster'])
print("Silhouette Score:", silhouette_avg)
```

---

### **3. Davies-Bouldin Index (DBI)**
- Measures the **average similarity** between clusters.
- **Lower values indicate better-separated clusters**.

```python
from sklearn.metrics import davies_bouldin_score

dbi = davies_bouldin_score(features_scaled, nba_df['Cluster'])
print("Davies-Bouldin Index:", dbi)
```

---

## **The Elbow Method: Choosing the Optimal K**
The **Elbow Method** helps determine the best **number of clusters (K)** by plotting **WCSS vs. K**.

```python
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()
```
![output (1)](https://github.com/user-attachments/assets/1d6435c7-20ff-4e86-973b-50b2711dd6dc)
![image](https://github.com/user-attachments/assets/a71e0533-995a-4351-9abf-5977ffd0a60c)

👉 **Interpretation:** The "elbow" point on the plot suggests the optimal number of clusters.

---

### **Summary**
- **K-Means** is an effective clustering algorithm that partitions data into **K clusters**.
- **Cluster quality** is evaluated using **WCSS, Silhouette Score, and DBI**.
- **The Elbow Method** helps determine the optimal **K value**.

Would you like me to generate a visualization for your dataset? 🚀

# **K-Means Clustering Overview**
---
K-Means is a popular **unsupervised machine learning algorithm** used for **clustering data into K groups**. It assigns data points to clusters based on their similarity, measured using **Euclidean distance**. The algorithm is widely used in customer segmentation, image recognition, and anomaly detection.

---
## What is Clustering?
**Cluster:** A group of similar data points based on certain features.

**K**: Number of groups specified in clustering algorithms. 

**Clustering Techniques** are among the most popular unsupervised machine learning algorithms. The main idea behind clustering is that you want to group objects into similar classes, in a way that:
- intra-class similarity is high (similarity amongst members of the same group is high) *(Example: Similarity between two players from the same team.)*
- inter-class similarity is low (similarity of different groups is low) *(Example: Differences between players from different teams.)*

**Why is Clustering Important?** Clustering groups similar data, helps in pattern discovery, and improves decision-making in various applications like recommendation systems, image segmentation, and market analysis.
- **Example:** In the NBA, clustering can group players based on playing style (e.g., shooters, defenders, playmakers), helping teams make better recruitment and game strategy decisions.

**How can you determine if they're similar?**: The closer two points are, the more similar they are. It is useful to make a distinction between hierarchical and nonhierarchical clustering algorithms:
**Agglomerative Hierarchical:**
1. It starts with **n** clusters (where n is the number of observations, so each ovservation is a cluster) -->
2. Then combines the two most similar clusters combines
3. Then combines the next two most similar cluster, and continues.
*(A divise one does the exact oppoiste going from 1 to n clusters)*
   
**Agglomerative Nonhierarchical**
1. Chooses **k** intitial cluster (Refers to k starting points, centroids, chosen in k-mean clustering to begin grouping data points) and reassign 

You basically try to group data together without knowing what actual cluster/classes are.

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



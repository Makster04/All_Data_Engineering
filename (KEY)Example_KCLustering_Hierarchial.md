# Finance Examples (K-Clustering Mean vs Hierarchial)
---

## 1. K-Means Clustering Example: Portfolio Segmentation

### Objective:
An investment management firm wants to categorize various stocks based on similar financial characteristics, such as volatility (standard deviation of returns) and returns (annualized returns), to efficiently diversify portfolios.

### Steps:

1. **Step 1 (Data Collection):** Collect stock performance data:

- Annual return percentage
- Volatility (standard deviation)
- Dividend yield

2. **Step 2 (Data Preparation):** Normalize these data points to make comparisons consistent.

| Stock | Return (%) | Volatility (%) | Dividend Yield (%) |
|-------|------------|----------------|---------------------|
| AAPL  | 12%        | 15%            | 0.7%                |
| MSFT  | 10%        | 14%            | 1.0%                |
| TSLA  | 25%        | 40%            | 0.0%                |
| PG    | 8%         | 12%            | 2.8%                |
| KO    | 6%         | 10%            | 3.0%                |

3. **Step 3 (Apply K-Means Clustering):** Using the normalized data, group stocks into clusters (e.g., 2 clusters):

**Cluster 1 (High-Risk, High-Return):**
- Tesla (TSLA)

**Cluster 2 (Stable, Moderate-Return):**
- Apple (AAPL)
- Microsoft (MSFT)
- Procter & Gamble (PG)
- Coca-Cola (KO)

### Interpretation:
- **Cluster 1**: Suitable for aggressive investors seeking high growth and willing to accept high volatility.
- **Cluster 2**: Suitable for conservative or dividend-focused investors looking for stability.

---

## 2. Hierarchical Clustering Example: Customer Segmentation in Banking

### Objective:
A retail bank wants to segment customers based on their financial behavior (e.g., monthly account balances, number of transactions per month, loan utilization) to tailor personalized financial products.

### Steps:

1. **Step 1 (Data Collection):** Gather customer account behavior:
- Average Monthly Account Balance
- Monthly Transactions Count
- Loan Utilization (percentage of total credit limit used)

2. **Step 2 (Data Preparation):** Normalize data for comparability:

| Customer ID | Avg Balance ($) | Transactions | Loan Utilization (%) |
|-------------|-----------------|--------------|----------------------|
| 101         | 15,000          | 30           | 10%                  |
| 102         | 3,000           | 5            | 80%                  |
| 103         | 25,000          | 40           | 5%                   |
| 104         | 1,000           | 3            | 90%                  |
| 105         | 10,000          | 20           | 20%                  |

3. **Step 3 (Apply Hierarchical Clustering):** Use Ward’s linkage or average linkage to create a dendrogram, grouping customers by similarity:

```
Level 1 Cluster:
├── Level 2 Cluster (High balances, frequent transactions, low loan use)
│   ├── Customer 101
│   ├── Customer 103
│   └── Customer 105
│
└── Level 2 Cluster (Low balances, fewer transactions, high loan use)
    ├── Customer 102
    └── Customer 104
```

### Interpretation:
- **High Balance Group (Cluster 1)**: Customers suitable for premium products (e.g., wealth management, investment advisory).
- **High Credit Utilization Group (Cluster 2)**: Customers suitable for products focused on debt restructuring, credit counseling, or higher credit offerings.

---

### Key Differences in Application:

| Criteria                  | K-Means Clustering                      | Hierarchical Clustering                    |
|---------------------------|-----------------------------------------|--------------------------------------------|
| **Clustering Approach**   | Iterative partitioning                  | Agglomerative (bottom-up) or divisive (top-down) |
| **Data Size**             | Suitable for large datasets             | Usually suitable for smaller datasets      |
| **Cluster Determination** | Pre-specify number of clusters (`k`)    | Determine clusters post-analysis via dendrogram |
| **Interpretation**        | Simple, clearly defined clusters        | Complex nested clusters; visual dendrogram |

Both methods are valuable, depending on your specific finance scenario, data availability, and the required complexity of interpretation.

---

Below are clear descriptions of visualization outputs you'd typically get from applying **K-Means** and **Hierarchical Clustering** to the given finance examples:

---
# Python Examples
---

# 1. **K-Means Clustering Visualization**

### **Scenario:** Portfolio Segmentation (Stocks grouped by Volatility vs. Returns)

### **Visualization: Scatter Plot (Clusters clearly separated by color)**

- **X-axis:** Annualized Return (%)
- **Y-axis:** Volatility (%)
- **Clusters:** Represented by color (e.g., Red, Blue, Green)
- **Centroids:** Represented by 'X' or bold markers at cluster centers.

Example Plot:

```
Volatility (%)
   ^
40 |                     (TSLA) [High Risk Cluster 🔴]
   |
30 |
   |
20 |
   |                      🔵(AAPL)
   |                      🔵(MSFT)
10 |       🔵(KO)   🔵(PG)   (Stable Cluster 🔵)
   |
   +------------------------------------------------> Return (%)
        5%       10%      15%       20%      25%     
```

### **Interpretation:**
- Clearly identifies "High Risk–High Return" stocks (TSLA) distinctly separated from "Stable" stocks (AAPL, MSFT, KO, PG).
- Investors can visually choose investments aligning with their risk preferences.

---

# 2. **Hierarchical Clustering Visualization**

### **Scenario:** Banking Customer Segmentation based on financial behavior (Balances, Transactions, Loans)

### **Visualization: Dendrogram (Tree-like structure)**

A dendrogram shows the hierarchical relationship between customers.

Example Dendrogram:

```
Height (Distance between clusters)
   ^
   |                     
   |                   ┌──── Customer 101
   |               ┌───┤
   |           ┌───┤   └──── Customer 103
   |           │   └──────── Customer 105
   |──────┤
   |           │       ┌──── Customer 102
   |           └───┤
   |                   └──── Customer 104
   +-------------------------------------------------------> Customers
```

### **Interpretation:**
- Clearly shows how customers naturally group based on similar financial behaviors.
- High balance, transaction-heavy customers grouped distinctly from lower balance, high loan-utilization customers.
- Helps banks quickly identify customer segments for targeted marketing and product offerings.

---

# **Real-Life Implementation using Python (Example)**

### **1. K-Means Clustering:**
```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Sample data: Returns and Volatility
X = np.array([[12,15], [10,14], [25,40], [8,12], [6,10]])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.labels_

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=150)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='X', s=200)
plt.xlabel('Returns (%)')
plt.ylabel('Volatility (%)')
plt.title('K-Means Clustering: Portfolio Segmentation')
plt.grid(True)
plt.show()
```

### **Output:**
- A scatter plot with clearly marked clusters and centroids.

---

### **2. Hierarchical Clustering:**
```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Sample data: Avg Balance, Transactions, Loan Utilization
X = np.array([[15000,30,10], [3000,5,80], [25000,40,5], [1000,3,90], [10000,20,20]])

linked = linkage(X, 'ward')

labelList = ['Customer 101','Customer 102','Customer 103','Customer 104','Customer 105']

plt.figure(figsize=(8,5))
dendrogram(linked,
           orientation='top',
           labels=labelList,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering: Customer Segmentation')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.grid(True)
plt.show()
```

### **Output:**
- A dendrogram clearly showing hierarchical groupings of customers.

---

These visualizations provide intuitive, actionable insights from both clustering methods in finance contexts.

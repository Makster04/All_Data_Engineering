# Finance Examples (K-Clustering Mean vs Hierarchial)
---

## **1. K-Means Clustering Example: Portfolio Segmentation**

### **Objective:**
An investment management firm wants to categorize stocks based on similar financial characteristics, such as volatility and returns, to efficiently diversify portfolios.

### **Steps:**

**Step 1: Data Collection**  
Collect stock performance data:
- Annual return (%)
- Volatility (%)
- Dividend yield (%)

**Step 2: Data Preparation**  
Normalize these data points for consistency.

| Stock | Return (%) | Volatility (%) | Dividend Yield (%) |
|-------|------------|----------------|---------------------|
| AAPL  | 12%        | 15%            | 0.7%                |
| MSFT  | 10%        | 14%            | 1.0%                |
| TSLA  | 25%        | 40%            | 0.0%                |
| PG    | 8%         | 12%            | 2.8%                |
| KO    | 6%         | 10%            | 3.0%                |

**Step 3: Apply K-Means Clustering**  
Using normalized data, group stocks into clusters (e.g., 2 clusters):

- **Cluster 1 (High-Risk, High-Return):**
  - Tesla (TSLA)

- **Cluster 2 (Stable, Moderate-Return):**
  - Apple (AAPL)
  - Microsoft (MSFT)
  - Procter & Gamble (PG)
  - Coca-Cola (KO)

### **Interpretation:**
- **Cluster 1**: Suitable for aggressive investors seeking high growth and accepting higher volatility.
- **Cluster 2**: Suitable for conservative or dividend-focused investors seeking stability.

### **Python Implementation (K-Means):**
```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Stock data: Returns and Volatility
X = np.array([[12,15], [10,14], [25,40], [8,12], [6,10]])

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Visualization
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=150)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='X', s=200)
plt.xlabel('Returns (%)')
plt.ylabel('Volatility (%)')
plt.title('K-Means Clustering: Portfolio Segmentation')
plt.grid(True)
plt.show()
```

### **Visualization:**
**Scatter Plot (Clusters clearly separated by color)**  
- X-axis: Annualized Return (%)
- Y-axis: Volatility (%)

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

---

## **2. Hierarchical Clustering Example: Customer Segmentation in Banking**

### **Objective:**
A retail bank aims to segment customers based on financial behaviors (monthly balances, transactions, loan utilization) to tailor personalized financial products.

### **Steps:**

**Step 1: Data Collection**  
Gather customer account behavior data:
- Average Monthly Account Balance
- Monthly Transactions Count
- Loan Utilization (% of total credit limit used)

**Step 2: Data Preparation**  
Normalize data for comparability:

| Customer ID | Avg Balance ($) | Transactions | Loan Utilization (%) |
|-------------|-----------------|--------------|----------------------|
| 101         | 15,000          | 30           | 10%                  |
| 102         | 3,000           | 5            | 80%                  |
| 103         | 25,000          | 40           | 5%                   |
| 104         | 1,000           | 3            | 90%                  |
| 105         | 10,000          | 20           | 20%                  |

**Step 3: Apply Hierarchical Clustering**  
Use Ward’s linkage to create a dendrogram, grouping customers by similarity:

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

### **Interpretation:**
- **High Balance Group (Cluster 1)**: Suitable for premium financial products (e.g., wealth management).
- **High Credit Utilization Group (Cluster 2)**: Suitable for debt management, credit counseling, or tailored credit products.

### **Python Implementation (Hierarchical):**
```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Customer behavior data
X = np.array([
    [15000, 30, 10],
    [3000, 5, 80],
    [25000, 40, 5],
    [1000, 3, 90],
    [10000, 20, 15]
])

linked = linkage(X, method='ward')

labelList = ['Customer 101','Customer 102','Customer 103','Customer 104','Customer 105']

# Dendrogram Visualization
plt.figure(figsize=(8,5))
dendrogram(linked := linkage(X, 'ward'),
           orientation='top',
           labels=labelList := ['Customer 101', 'Customer 102', 'Customer 103', 'Customer 104', 'Customer 105'],
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering: Customer Segmentation')
plt.show()
```

### **Visualization (Dendrogram):**
```
Height (Distance between clusters)
   |
   ├── Level 2 Cluster (High balances, frequent transactions, low loan use)
   │   ├── Customer 101
   │   ├── Customer 103
   │   └── Customer 105
   │
   └── Level 2 Cluster (Low balances, fewer transactions, high loan use)
       ├── Customer 102
       └── Customer 104
```

### **Interpretation:**
- Clearly indicates how customers naturally group based on similar financial behaviors.
- Allows easy identification of customer segments suitable for tailored financial services.

---

## **Key Differences (Summary):**

| Criteria              | K-Means Clustering                | Hierarchical Clustering                             |
|-----------------------|------------------------------------|----------------------------------------------|
| Clustering Approach   | Iterative partitioning             | Agglomerative (bottom-up) or divisive (top-down) |
| Data Size             | Suitable for large datasets        | Usually better for smaller datasets          |
| Visualization         | Scatter plot (clearly defined clusters) | Dendrogram (shows hierarchy clearly)           |

---

This structure organizes the theory, data preparation, interpretation, and Python examples clearly within each clustering example for ease of understanding.

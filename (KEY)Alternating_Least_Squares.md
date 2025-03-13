# **Matrix Factorization with Alternating Least Squares (ALS)**

Matrix Factorization is a core technique in **collaborative filtering-based recommender systems**. One of the most effective methods is **Alternating Least Squares (ALS)**, which factorizes a user-item interaction matrix into **latent factors** that represent hidden relationships.

---

## **1. What is Alternating Least Squares (ALS)?**
ALS is a matrix factorization technique that decomposes a **user-item rating matrix $$\( R \)$$** into two lower-dimensional matrices:

$$\[
R \approx U \times V^T
\]$$

where:
- $$\( U \)$$ is the **user feature matrix** ($$\( m \times k \)$$)
- $$\( V \)$$ is the **item feature matrix** ($$\( n \times k \)$$)
- $$\( k \)$$ is the number of **latent factors** that capture user and item preferences.

Instead of solving for **$$\( U \)$$ and $$\( V \)$$ simultaneously**, ALS **alternates** between solving for one while keeping the other fixed, making the optimization problem computationally feasible.

---

## **2. How ALS Works**
1. **Initialize $$\( U \)$$ and $$\( V \)$$** with random values.
2. **Fix $$\( V \)$$, solve for $$\( U \)$$** by minimizing the least squares error.
3. **Fix $$\( U \)$$, solve for $$\( V \)$$** by minimizing the least squares error.
4. **Repeat** until convergence.

Each step solves a **least squares regression problem**, making ALS an efficient method that can be **parallelized**.

---

## **3. ALS and Matrix Decomposition**
ALS is closely related to **matrix decomposition** because it **factorizes a large, sparse matrix** into two dense matrices that capture patterns. Unlike **Singular Value Decomposition (SVD)**, which requires dense matrices, ALS can handle **sparse data** efficiently.

### **Why ALS Works Well with Sparse Data**
- **Missing values**: Instead of filling missing values with zeros (as in SVD), ALS optimizes only for known interactions.
- **Regularization**: ALS incorporates **L2 regularization** to prevent overfitting.
- **Scalability**: ALS is easy to distribute across multiple nodes.

---

## **4. Why ALS Can Be Parallelized Efficiently**
Each step in ALS solves a **linear least squares problem**, which involves computing:

$$\[
U_i = (V^T V + \lambda I)^{-1} V^T R_i
\]$$

Since each row of $$\( U \)$$ (or $$\( V \)$$) can be updated **independently**, ALS is **highly parallelizable** across multiple CPUs or GPUs.

### **Parallelization Benefits**
- Each **user**'s preferences are updated **independently**.
- Each **item**'s features are updated **independently**.
- Works well in distributed computing environments like **Apache Spark's MLlib**.

---

## **5. Incorporating Bias Terms for More Accurate Embeddings**
A basic ALS model assumes all interactions are based solely on latent factors. However, **bias terms** can improve accuracy by capturing inherent **user and item tendencies**.

### **Extended ALS Model with Bias Terms**
The predicted rating \( \hat{R}_{ij} \) is:

$$\[
\hat{R}_{ij} = \mu + b_i + b_j + U_i \cdot V_j^T
\]$$

where:
- $$\( \mu \)$$ is the **global average rating**.
- $$\( b_i \)$$ is the **user bias** (e.g., some users rate generously).
- $$\( b_j \)$$ is the **item bias** (e.g., some movies are generally rated higher).
- $$\( U_i \cdot V_j^T \)$$ is the interaction between user and item latent factors.

Bias terms help when:
- Some users rate **consistently high or low**.
- Some items have **higher/lower average ratings**.

---

## **6. Example: Implementing ALS in Python**
Let's implement **ALS-based matrix factorization** using **NumPy**.

### **Step 1: Create a Simulated User-Item Matrix**
```python
import numpy as np

# Simulated User-Item Matrix (Ratings)
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 4, 3],
    [1, 1, 0, 5],
    [0, 3, 4, 4]
], dtype=np.float32)

# Define parameters
num_users, num_items = R.shape
num_factors = 2  # Latent factors
lambda_reg = 0.1  # Regularization parameter
num_iterations = 10
```

---

### **Step 2: Implement ALS**
```python
# Initialize user and item matrices with random values
U = np.random.rand(num_users, num_factors)
V = np.random.rand(num_items, num_factors)

# ALS Iterative Optimization
for iteration in range(num_iterations):
    # Solve for U while keeping V fixed
    for i in range(num_users):
        idx = R[i, :] > 0  # Select rated items
        V_sub = V[idx]  # Corresponding item features
        R_sub = R[i, idx]  # User's actual ratings
        U[i] = np.linalg.solve(V_sub.T @ V_sub + lambda_reg * np.eye(num_factors), V_sub.T @ R_sub)

    # Solve for V while keeping U fixed
    for j in range(num_items):
        idx = R[:, j] > 0  # Select rated users
        U_sub = U[idx]  # Corresponding user features
        R_sub = R[idx, j]  # Item's actual ratings
        V[j] = np.linalg.solve(U_sub.T @ U_sub + lambda_reg * np.eye(num_factors), U_sub.T @ R_sub)

# Compute the final predicted ratings
predicted_R = U @ V.T

# Display the predicted rating matrix
import pandas as pd
predicted_df = pd.DataFrame(predicted_R, columns=[f"Movie{j+1}" for j in range(num_items)], index=[f"User{i+1}" for i in range(num_users)])
predicted_df
```

---

## **7. Summary**
- **ALS is a powerful matrix factorization technique** that efficiently learns latent features.
- **Bias terms improve accuracy** by capturing user/item tendencies.
- **ALS can handle sparse data well** and is easily parallelizable.
- **Unlike SVD, ALS does not require dense matrices**, making it better for large-scale recommendations.

Would you like a real-world dataset example (e.g., MovieLens) using **Apache Spark ALS**? 🚀

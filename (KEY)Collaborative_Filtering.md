# **Collaborative Filtering with Singular Value Decomposition (SVD)**
---
**Collaborative Filtering (CF)** is a popular approach in recommendation systems where user preferences are analyzed to suggest relevant items. 

- **Singular Value Decomposition (SVD)** is a powerful matrix factorization technique used in **model-based collaborative filtering** to extract latent features from a user-item interaction matrix.

---

## **1. Memory-Based vs. Model-Based Collaborative Filtering**

| Feature | Memory-Based CF | Model-Based CF |
|---------|---------------|---------------|
| **Approach** | Uses similarity measures (e.g., Pearson, Cosine) between users or items | Uses machine learning models and matrix factorization (e.g., SVD, Autoencoders) |
| **Computation** | Requires storing and searching for similarity in a large dataset | Learns patterns and generalizes with fewer computations at runtime |
| **Sparsity Handling** | Performs poorly with sparse data due to missing interactions | Handles sparsity well by learning latent factors |
| **Scalability** | Slow and computationally expensive for large datasets | More efficient after training, as predictions can be made quickly |
| **Cold Start Problem** | Struggles with new users/items due to lack of interactions | Mitigates cold start by learning general patterns |

---

## **2. How Memory-Based Collaborative Filtering Works**
Memory-based collaborative filtering is divided into two types:

### **A. User-User Collaborative Filtering**
- Measures similarity between users based on their past interactions.
- Example: If Alice and Bob have similar movie ratings, a movie liked by Alice is recommended to Bob.

**Similarity Metrics:**
- **Cosine Similarity**: Measures the cosine of the angle between two vectors.
- **Pearson Correlation**: Measures the linear correlation between two users.
- **Jaccard Similarity**: Measures overlap between sets of items.

🔹 **Disadvantage:** Requires the entire dataset in memory, leading to slow performance for large datasets.

### **B. Item-Item Collaborative Filtering**
- Finds similar items instead of similar users.
- Example: If many users watch **Inception** and **Interstellar**, someone who watches **Inception** is recommended **Interstellar**.

🔹 **Advantage:** More stable recommendations as user preferences change over time.

---

## **3. How Model-Based Collaborative Filtering Works**
Model-based CF applies machine learning or matrix factorization techniques to learn user preferences.

### **A. Matrix Factorization (SVD, ALS, NMF)**
Instead of storing all user-item interactions, matrix factorization techniques decompose the large matrix into smaller ones, extracting **latent factors** that represent hidden patterns.

---

## **4. How SVD Extracts Meaning with Latent Factors**
SVD decomposes a **user-item matrix (R)** into three matrices:

$$\[
R = U \Sigma V^T
\]$$

Where:
- $$\( U \)$$ (Users × Latent Factors) represents users in a lower-dimensional space.
- $$\( \Sigma \)$$ (Latent Factors × Latent Factors) is a diagonal matrix with singular values.
- $$\( V^T \)$$ (Items × Latent Factors) represents items in the same lower-dimensional space.

🔹 **How it works in recommendation:**
1. **Reduces dimensionality**: Captures important information while filtering noise.
2. **Finds latent factors**: Discovers abstract relationships (e.g., “sci-fi lovers” as a hidden preference).
3. **Makes predictions**: Reconstructs missing values in the original matrix to predict ratings.

---

## **5. Implementing SVD Using SciPy**
Let's use **SVD** from `scipy.linalg` to factorize a user-item rating matrix and predict missing values.

```python
import numpy as np
from scipy.linalg import svd

# Sample User-Item Rating Matrix (Rows: Users, Columns: Items)
ratings_matrix = np.array([
    [5, 4, 0, 1],
    [4, 0, 4, 3],
    [3, 5, 3, 0],
    [0, 3, 5, 4]
])

# Apply Singular Value Decomposition
U, S, Vt = svd(ratings_matrix, full_matrices=False)

# Convert singular values into a diagonal matrix
S_diag = np.diag(S)

# Reconstruct the matrix
reconstructed_matrix = np.dot(U, np.dot(S_diag, Vt))

# Display results
import ace_tools as tools
import pandas as pd

df = pd.DataFrame(reconstructed_matrix, columns=['Item1', 'Item2', 'Item3', 'Item4'])

# Display the predicted ratings dataframe
predicted_ratings_df.head()
```
```
         Movie1    Movie2    Movie3    Movie4    Movie5    Movie6
User1  4.741722  3.489687  0.216269  1.041723 -0.337627  3.747507
User2  2.177032  0.702052  2.223951  3.758466  2.727082  1.610015
User3  3.240742  5.548790  2.799675  0.033331  0.250812  4.209246
User4  0.735292  2.818951  4.214667  2.108890  2.676652  2.050386
User5  3.193100 -0.264542  1.364048  4.918506  2.950328  1.586077

```

---

## **Key Takeaways**
1. **Memory-based CF** relies on user/item similarity but struggles with large datasets.
2. **Model-based CF** (e.g., SVD) generalizes better by finding **latent factors**.
3. **SVD decomposes the user-item matrix**, capturing underlying structures in user preferences.
4. **SciPy’s SVD implementation** allows reconstructing missing values for better recommendations.

Would you like to extend this with a real-world dataset like MovieLens? 🚀

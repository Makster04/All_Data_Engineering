# Step-by-Step using KNN algorithm
---

## 📌 **Step-by-Step Explanation**

### 1. **Choose a data point for which you want to predict a label:**
- Imagine you have a dataset of labeled points (some have label A, others have label B), and you want to predict the label of a new, unknown point.

### 2. **Identify the K nearest points around this new point:**
- Choose a value for **K**. Typical choices are small, odd numbers like 1, 3, 5, or 11.
- Calculate the **distance** from your chosen point to all other points. Typically, Euclidean distance is used:
\[
\text{Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + \dots}
\]

### 3. **Predict the label based on these K neighbors:**
- **Classification**: Assign the most common class label among the K neighbors.
- **Regression**: Calculate the average value of the target variable among the K neighbors.

### 4. **(Optional) Use weighted averages:**
- Instead of treating each neighbor equally, assign weights based on distance (closer neighbors have more influence):
  \[
  \text{Weighted Average} = \frac{\sum (\text{value of neighbor}_i \times \frac{1}{\text{distance}_i})}{\sum (\frac{1}{\text{distance}_i})}
  \]

---

## 🚀 **Example Illustration (Classification)**:

Suppose you have the following points (two classes: ⚪️ and 🔵):

```
(1, 2) 🔵 
(2, 3) 🔵
(3, 3) 🔵
(6, 5) ⚪️
(7, 7) ⚪️
(8, 6) ⚪️
```

Now you have a new point:  
```
(5, 4) ❓
```

Let's find its class using **K = 3**.

### Step-by-Step Solution:

**1.** Calculate distance to all points:

| Existing Point | Class | Distance to (5,4)                |
|----------------|-------|----------------------------------|
| (1,2)          | 🔵    | √((5-1)²+(4-2)²)= √(16+4)=√20 ≈4.47|
| (2,3)          | 🔵    | √((5-2)²+(4-3)²)= √(9+1)=√10 ≈3.16 |
| (3,3)          | 🔵    | √((5-3)²+(4-3)²)= √(4+1)=√5 ≈2.24  |
| (6,5)          | ⚪️    | √((5-6)²+(4-5)²)= √(1+1)=√2 ≈1.41  |
| (7,7)          | ⚪️    | √((5-7)²+(4-7)²)= √(4+9)=√13≈3.61 |
| (8,6)          | ⚪️    | √((5-8)²+(4-6)²)= √(9+4)=√13≈3.61 |

**2.** Choose the 3 nearest neighbors:
- (6,5) ⚪️ Distance=1.41  
- (3,3) 🔵 Distance=2.24  
- (2,3) 🔵 Distance=3.16  

**3.** Majority voting among neighbors:
- 🔵 = 2 votes
- ⚪️ = 1 vote

**Prediction:**  
```
(5,4) → 🔵
```

---

## 📐 **Example Illustration (Regression)**:

Now imagine you have a regression problem, and your neighbors have numeric values instead:

| Existing Point | Value | Distance to (5,4)|
|----------------|-------|-------------------|
| (1,2)          | 100   | ≈4.47             |
| (2,3)          | 90    | ≈3.16             |
| (3,3)          | 80    | ≈2.24             |
| (6,5)          | 50    | ≈1.41             |
| (7,7)          | 30    | ≈3.61             |
| (8,6)          | 20    | ≈3.61             |

Again, choose **K=3** neighbors closest to (5,4):

- (6,5): value=50
- (3,3): value=80
- (2,3): value=90

Average these values:
\[
\text{Prediction} = \frac{50 + 80 + 90}{3} = \frac{220}{3} ≈ 73.33
\]

---

## 🎯 **Using Weighted Averages**:

You could instead weight by distance (closer neighbors matter more):

\[
\text{Prediction} = \frac{\frac{50}{1.41} + \frac{80}{2.24} + \frac{90}{3.16}}{\frac{1}{1.41} + \frac{1}{2.24} + \frac{1}{3.16}} 
\]

This gives greater emphasis to closer neighbors.

---

## ⚖️ **When to Use KNN:**

- **Pros:**
  - Simple and intuitive
  - No assumptions about underlying data distribution
  - Works well for multi-class classification
  
- **Cons:**
  - Computationally intensive with large datasets
  - Sensitive to irrelevant features and scaling

---

That's how the **K-Nearest Neighbors (KNN)** algorithm works in practice!

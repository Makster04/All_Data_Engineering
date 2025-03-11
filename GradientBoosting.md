Gradient Boosting is a machine learning technique that builds an ensemble of weak learners (typically decision trees) in a sequential manner, where each new model corrects the errors of the previous one. Here’s a step-by-step breakdown:

### **Steps in Gradient Boosting:**
1. **Initialize the Model**  
   - Start with a simple model, usually a constant value (like the mean of the target variable).
   
2. **Compute Residuals (Errors)**  
   - Calculate the difference (residuals) between the actual values and the predictions from the initial model.

3. **Train a Weak Learner (Decision Tree) on Residuals**  
   - Fit a small decision tree (weak learner) to predict these residuals.

4. **Update the Model**  
   - Add the new tree’s predictions to the existing model with a learning rate to control its impact.

5. **Repeat Steps 2-4**  
   - Continue adding trees iteratively until a stopping criterion is met (e.g., maximum trees, minimal improvement).

6. **Final Prediction**  
   - The final model is a sum of all trees’ weighted predictions.

### **Mathematical Representation**
Given a dataset with features \( X \) and target \( y \), we initialize:
\[
F_0(X) = \text{mean}(y)
\]
At each iteration \( m \), we compute residuals:
\[
r_i^{(m)} = y_i - F_{m-1}(X_i)
\]
Then, fit a new decision tree \( h_m(X) \) to predict \( r_i^{(m)} \), and update the model:
\[
F_m(X) = F_{m-1}(X) + \eta h_m(X)
\]
where \( \eta \) is the learning rate.

---

### **Implementation in Python using Scikit-Learn**
Let's demonstrate Gradient Boosting using `sklearn` on a sample dataset.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=5, noise=0.2, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

### **Key Hyperparameters in Gradient Boosting**
- `n_estimators`: Number of trees (iterations).
- `learning_rate`: Controls the contribution of each tree (smaller values need more trees).
- `max_depth`: Depth of each weak learner (tree complexity).
- `subsample`: Fraction of samples used for training each tree (introduces randomness).
- `min_samples_split`: Minimum samples required to split a node.

---

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import ace_tools as tools

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=5, noise=0.2, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Get feature importance
feature_importance = gb_model.feature_importances_
feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(feature_names, feature_importance, color='blue', edgecolor='black')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Gradient Boosting")
plt.gca().invert_yaxis()  # Highest importance on top

# Show plot
plt.show()
```
![image](https://github.com/user-attachments/assets/86507e63-38b9-4cac-bbcf-629ab3072050)



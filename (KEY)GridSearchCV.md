# Steps for GridSearchCV

Here's a step-by-step guide on how to design a parameter grid for `GridSearchCV` and use it to improve model performance in `scikit-learn`.

---

### **Step 1: Import Required Libraries**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

---

### **Step 2: Load and Split the Dataset**
```python
# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
---

### **Step 3: Standardize your Data**
```python
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data (using the same scaler)
X_test_scaled = scaler.transform(X_test)
```

---

### **Step 4: Define a Parameter Grid**
```python
# Step 4: Define a Parameter Grid for KNeighborsClassifier
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
```

---

### Step 5: Initialize and Perform Grid Search
```python
knn = KNeighborsClassifier()

grid_search = GridSearchCV(
    estimator=knn, 
    param_grid=param_grid, 
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available processors
    verbose=2  # Display progress
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)
```

---

### **Step 6: Evaluate the Best Model**
```python
# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
best_params, accuracy
```
```
Fitting 5 folds for each of 24 candidates, totalling 120 fits
```

---

### **Why Use GridSearchCV?**
- **Automates hyperparameter tuning:** Tries different parameter combinations systematically.
- **Cross-validation reduces overfitting:** Ensures the model generalizes well.
- **Optimizes performance:** Helps find the best parameters for maximum accuracy.

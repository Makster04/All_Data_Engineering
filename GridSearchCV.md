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

### **Step 3: Define a Parameter Grid**
```python
# Define the parameter grid for tuning RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],        # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],        # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],          # Minimum samples required at each leaf node
    'bootstrap': [True, False]              # Whether bootstrap samples are used
}
```

---

### **Step 4: Initialize and Perform Grid Search**
```python
# Initialize the classifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    cv=5,           # 5-fold cross-validation
    n_jobs=-1,      # Use all available processors
    verbose=2       # Display progress
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)
```

---

### **Step 5: Evaluate the Best Model**
```python
# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"Best Parameters: {best_params}")
print(f"Test Set Accuracy: {accuracy:.4f}")
```

---

### **Why Use GridSearchCV?**
- **Automates hyperparameter tuning:** Tries different parameter combinations systematically.
- **Cross-validation reduces overfitting:** Ensures the model generalizes well.
- **Optimizes performance:** Helps find the best parameters for maximum accuracy.

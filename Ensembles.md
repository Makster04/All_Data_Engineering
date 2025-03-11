# Ensembles methods

Here are Python coding examples for each of the topics:

---

### **1. Random Forests**
Random forests create multiple decision trees and aggregate their results for better accuracy.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy:.2f}')
```

---

### **2. GridSearchCV for Hyperparameter Tuning**
GridSearchCV helps find the best hyperparameters for decision trees.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define hyperparameters to tune
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize model
dt = DecisionTreeClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Tuned Decision Tree Accuracy: {accuracy:.2f}')
```

---

### **3. Gradient Boosting and Weak Learners**
Gradient Boosting sequentially builds trees, correcting previous errors.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Create and train a Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Gradient Boosting Accuracy: {accuracy:.2f}')
```

---

### **4. XGBoost (Extreme Gradient Boosting)**
XGBoost is an optimized version of Gradient Boosting.

```python
import xgboost as xgb

# Create and train an XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'XGBoost Accuracy: {accuracy:.2f}')
```

---

Each of these models improves on decision trees in different ways:
- **Random Forests** reduce overfitting by averaging multiple decision trees.
- **GridSearchCV** optimizes model performance by tuning hyperparameters.
- **Gradient Boosting** focuses on correcting mistakes from previous weak learners.
- **XGBoost** is an optimized version of Gradient Boosting with better speed and performance.

Let me know if you need modifications or explanations! 🚀

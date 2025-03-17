# Pipelines
---
Sure! In **Machine Learning**, pipelines are used to **automate workflows** by chaining together different data processing steps, feature transformations, and model training in a structured sequence.

### **Why Use Pipelines?**
- Ensures **consistent** preprocessing and transformation.
- Avoids **data leakage** by applying transformations only to training data before testing.
- Makes the code **modular** and easy to manage.
- Helps in **hyperparameter tuning** with GridSearchCV or RandomizedSearchCV.

---

### **Example: Machine Learning Pipeline with Scikit-Learn**
Here’s an example where we:
1. **Load the dataset** (Iris dataset)
2. **Split the dataset** into training and testing sets
3. **Create a pipeline** that includes:
   - Standard scaling (to normalize features)
   - Applying a classifier (Random Forest)

#### **Implementation in Python**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Normalize data
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Step 2: Classifier
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Pipeline Model Accuracy: {accuracy:.2f}")
```

---

### **Advanced: Adding Feature Selection and Hyperparameter Tuning**
We can extend pipelines to include:
- **Feature Selection**: The process of choosing the most important variables from a dataset to improve model performance, reduce complexity, and prevent overfitting by removing irrelevant or redundant features.
- **Hyperparameter tuning using GridSearchCV**: Method to find the best model settings by testing multiple hyperparameter combinations, evaluating performance using cross-validation, and selecting the optimal configuration to improve accuracy and generalization.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

# Create extended pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif)),  # Feature selection
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid for tuning
param_grid = {
    'feature_selection__k': [2, 3, 4],  # Select top k features
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20]
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
```

---

### **Key Takeaways**
1. **Pipelines automate the workflow** by ensuring all steps are applied consistently.
2. **Preprocessing steps** like feature scaling and selection are part of the pipeline.
3. **Hyperparameter tuning** can be integrated using `GridSearchCV`.
4. **Reduces code duplication** and improves reproducibility.


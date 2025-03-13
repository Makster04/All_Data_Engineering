# DSC Implementing Recommender Systems
---
The Surprise library is a powerful Python tool for building and evaluating recommender systems. It streamlines tasks like loading data, selecting algorithms, evaluating their performance, tuning hyperparameters, and making predictions. Below is an explanation of the key components and steps, along with code examples and important terms to help you understand how to implement recommendation engines using Surprise.

---

## Key Concepts and Terms

1. **Collaborative Filtering:** A method of making automatic predictions about user interests by collecting preferences from many users.
2. **Explicit Ratings vs. Implicit Feedback:** Explicit ratings (e.g., movie ratings from 1-5 stars) are commonly used, but implicit feedback (e.g., clicks, watch time) can also be modeled.
3. **User-Item Matrix:** A sparse matrix where each row represents a user, and each column represents an item, with values indicating ratings.
4. **Cold Start Problem:** When new users or items enter the system, there is not enough historical data to generate accurate recommendations.

- **Rating Scale:**  
  The range of ratings users can give items (e.g., 1 to 5). When processing your data with Surprise, you must specify this range using its built-in `Reader` class.

- **Reader Class:**  
  Surprise's `Reader` class helps parse your dataset (e.g., a CSV file). You provide it with the rating scale and data format so that Surprise can properly interpret the ratings.

- **Dataset and Trainset:**  
  - **Dataset:** Once loaded, your raw data becomes a `Dataset` object in Surprise.  
  - **Trainset:** This is the internal representation of the dataset used for training. It contains all the data points transformed for the algorithms.

- **Prediction:**  
  Once you train a model, you can obtain a prediction for a specific user-item pair using the `predict()` method.

- **Cross-Validation:**  
  Using the `cross_validate` function, you can assess the performance of your algorithms by splitting the dataset into multiple folds, training on a subset, and testing on the remaining data.

- **Algorithms:**  
  Surprise comes with several built-in algorithms such as:
  - **SVD (Singular Value Decomposition):** A matrix factorization method effective in uncovering latent features.
  - **KNN Variants:**  
    - **KNNBasic:** The most straightforward k-nearest neighbors approach.
    - **KNNWithMeans:** Similar to KNNBasic, but it also incorporates the mean rating of the users.
    - **KNNBaseline:** Incorporates baseline estimates to improve predictions.

- **Hyperparameter Tuning:**  
  Using `GridSearchCV`, you can search over a parameter grid to find the best set of hyperparameters for your recommender algorithm.

---

## Example Walkthrough (User-Item)

Below is a detailed example demonstrating how to:
1. **Load Data:** Use the `Reader` to read in your dataset.
2. **Train and Cross-Validate Models:** Use algorithms like SVD and KNN variants.
3. **Hyperparameter Tuning:** Use `GridSearchCV` to optimize your model.
4. **Predict for a Specific User-Item Pair:** Get a prediction after training.

```python
# Import necessary modules from Surprise and numpy
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise.prediction_algorithms import SVD, KNNBasic, KNNWithMeans, KNNBaseline
import numpy as np
```

### 1. Loading and Preparing Data
The `Reader` class is configured to match the CSV file format (with a defined rating scale of 1 to 5). The dataset is then loaded into Surprise using `Dataset.load_from_file()`.
```python
# Assume we have a CSV file 'ratings.csv' with the format: userID, itemID, rating, timestamp
# For example:
# 196,242,3,881250949
# 186,302,3,891717742
# ...
# Define the rating scale (here ratings are between 1 and 5)
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 5))

# Load the dataset from the file
data = Dataset.load_from_file('ratings.csv', reader=reader)
```
### 2. Cross-Validation with Different Algorithms
Using `cross_validate`, we evaluate the performance of the SVD and KNNBasic algorithms over 5 folds. This helps in understanding how well the algorithm generalizes.

```python
# Choose an algorithm, e.g., SVD
algo_svd = SVD()

# Perform 5-fold cross-validation on the SVD algorithm
cv_results = cross_validate(algo_svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("SVD Cross-validation results:")
print(cv_results)

# You can similarly evaluate other algorithms like KNNBasic, KNNWithMeans, and KNNBaseline
algo_knn = KNNBasic()
cv_results_knn = cross_validate(algo_knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("KNNBasic Cross-validation results:")
print(cv_results_knn)
```
### 3. Hyperparameter Tuning with GridSearchCV
`GridSearchCV` is used to search for the optimal parameters for SVD. The parameter grid includes variations in the number of latent factors, epochs, learning rate, and regularization. The best parameters and RMSE score are printed.
```python
# Define a parameter grid to search over for the SVD algorithm
param_grid = {
    'n_factors': [50, 100],  # number of latent factors
    'n_epochs': [20, 30],    # number of epochs for training
    'lr_all': [0.002, 0.005],# learning rate for all parameters
    'reg_all': [0.02, 0.1]   # regularization term for all parameters
}

# Set up grid search with SVD
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, joblib_verbose=2)
grid_search.fit(data)

# Get the best score and parameters
print("Best RMSE score obtained:", grid_search.best_score['rmse'])
print("Best parameters for SVD:", grid_search.best_params['rmse'])
```
### 4. Training on Full Trainset and Predicting a Rating
Once the best parameters are identified, the model is trained on the entire dataset (converted into a trainset). Finally, the `predict()` method is used to obtain the predicted rating for a specific user-item pair.

```python
# Build the full trainset from the dataset
trainset = data.build_full_trainset()

# Train the SVD algorithm with the best found parameters
best_svd = grid_search.best_estimator['rmse']
best_svd.fit(trainset)

# Predict the rating for a specific user and item
user_id = '196'   # Use the appropriate type (string or int) that matches your dataset
item_id = '302'
prediction = best_svd.predict(user_id, item_id)
print(f"Predicted rating for user {user_id} and item {item_id}: {prediction.est:.2f}")
```
---
## OUTPUT Explination:

### **1. Cross-validation Results**
After running `cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5)`, you will see results like:

```
SVD Cross-validation results:
{'fit_time': [0.23, 0.25, 0.24, 0.23, 0.26],
 'test_rmse': [0.92, 0.95, 0.91, 0.93, 0.94],
 'test_mae': [0.72, 0.74, 0.71, 0.73, 0.72]}
```

👉 **Interpretation:**
- **RMSE (Root Mean Square Error)**: Lower is better. It shows how much the predicted ratings deviate from the actual ratings.
- **MAE (Mean Absolute Error)**: Similar to RMSE but measures absolute errors instead of squared errors.
- The lower these values, the better the model. Usually, an RMSE below **1.0** is considered decent.

Similarly, for `KNNBasic`, you might get:
```
KNNBasic Cross-validation results:
{'fit_time': [0.35, 0.33, 0.32, 0.34, 0.31],
 'test_rmse': [1.02, 1.05, 1.00, 1.03, 1.01],
 'test_mae': [0.82, 0.85, 0.81, 0.84, 0.83]}
```
👉 **Interpretation:**
- Higher RMSE/MAE than `SVD` suggests `SVD` performs better on this dataset.

---

### **2. Hyperparameter Tuning Results**
After running `GridSearchCV`, you’ll see:

```
Best RMSE score obtained: 0.89
Best parameters for SVD: {'n_factors': 100, 'n_epochs': 30, 'lr_all': 0.005, 'reg_all': 0.02}
```

👉 **Interpretation:**
- **Best RMSE**: This is the lowest RMSE found during hyperparameter tuning.
- **Best Parameters**: These are the most optimal hyperparameters for `SVD`. You should use these values when training your final model.

---

### **3. Final Prediction**
After training the model with the best parameters, you will get:

```
Predicted rating for user 1 and item 102: 4.23
```

👉 **Interpretation:**
- If user **1** hasn’t rated item **102** before, this is the estimated rating the user would likely give.
- The model predicts the user will rate this item around **4.23** on a scale of **1-5**.

---

### **Conclusion**
- `SVD` performed better than `KNNBasic` based on RMSE and MAE.
- Hyperparameter tuning improved the RMSE from ~0.92 to **0.89**.
- The model can now make reasonable predictions for unseen user-item interactions.

---

## Important Points to Remember

- **Data Preparation:** Ensure your data is correctly formatted and the rating scale is specified correctly.
- **Evaluation Metrics:** RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) are common metrics to measure the performance of recommender systems.
- **Algorithm Choice:** Different algorithms (SVD, KNN variants) have different strengths. Experiment with multiple approaches.
- **Tuning Parameters:** Hyperparameter tuning (via `GridSearchCV`) can significantly improve the performance of your recommendation engine.
- **Predictions:** After training, use the `predict()` method to generate recommendations for any user-item pair.

This structured approach with Surprise helps in building robust recommendation systems by allowing you to experiment with different algorithms, evaluate them properly, and fine-tune them for the best performance.



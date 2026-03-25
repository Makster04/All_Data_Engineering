

---

# Machine Learning Concepts & Techniques

**Supervised Learning:** A machine learning approach where models are trained on labeled data, meaning the correct output is already known, so the model learns to predict future outcomes or classifications.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[3.5]]))

# The model learned from labeled examples.
# Smaller values were linked to class 0 and larger values to class 1.
# Since 3.5 is closer to the higher values, it predicts class 1.
[1]
```

**Unsupervised Learning:** A machine learning approach where models are trained on unlabeled data to identify hidden patterns, structures, or groupings without a known target variable.

```python
from sklearn.cluster import KMeans

X = [[1, 2], [1, 1], [8, 9], [9, 8]]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

print(kmeans.labels_)

# There are no labels like 0 or 1 given to the model.
# It groups points based on similarity and distance.
# The first two points are close together, and the last two are close together,
# so it places them into two different clusters.
[1 1 0 0]
```

**Cross-Validation:** A model evaluation technique that splits data into multiple parts, or folds, so the model can be trained and tested on different subsets to better measure how well it generalizes.

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=3)

print(scores)

# The dataset is split into 3 folds.
# The model trains on some folds and tests on the remaining fold,
# repeating this process 3 times.
# Each value shows the score for one fold.
[1. 1. 1.]
```

**Train-Test Split:** Dividing a dataset into separate training and testing portions so the model can be evaluated on unseen data.

```python
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print(X_train)
print(X_test)

# The data is split into two parts.
# 60% goes to training and 40% goes to testing.
# The exact rows depend on the random_state value.
[[3], [1], [4]]
[[2], [5]]
```

**Hyperparameter Tuning:** The process of finding the best model settings, such as tree depth or learning rate, to improve performance.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

grid = GridSearchCV(
    DecisionTreeClassifier(),
    {"max_depth": [1, 2, 3]},
    cv=3
)
grid.fit(X, y)

print(grid.best_params_)

# GridSearchCV tries each max_depth value.
# It uses cross-validation to see which setting performs best.
# The result shows the best hyperparameter found.
{'max_depth': 1}
```

**Decision Boundary:** The line, curve, or surface that separates different classes in a feature space and determines how a model classifies new data points.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [8], [9]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[2.5]]))
print(model.predict([[8.5]]))

# The model learns a dividing point between the two classes.
# 2.5 falls on the side associated with class 0.
# 8.5 falls on the side associated with class 1.
[0]
[1]
```

**Feature Engineering:** Creating new variables or transforming existing ones to improve model performance.

```python
import pandas as pd

df = pd.DataFrame({
    "income": [50000],
    "household_size": [2]
})

df["income_per_person"] = df["income"] / df["household_size"]
print(df)

# A new feature is created from existing columns.
# Instead of only income and household size,
# we now also have income per person, which may be more useful to a model.
   income  household_size  income_per_person
0   50000               2            25000.0
```

**Feature Selection:** Choosing the most relevant input variables to improve efficiency, interpretability, and model accuracy.

```python
from sklearn.feature_selection import SelectKBest, f_classif

X = [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]]
y = [0, 0, 1, 1]

selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

print(X_new)

# The method scores each feature based on how useful it is for predicting y.
# Since k=2, it keeps only the two best features.
# The output now has 2 columns instead of 3.
[[ 10 100]
 [ 20 200]
 [ 30 300]
 [ 40 400]]
```

**Model Deployment:** Making a trained machine learning model available for use in real applications.

```python
import pickle
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = DecisionTreeClassifier()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved")

# The trained model is written to a file.
# This lets you load and use it later without retraining it.
# Saving a model like this is one common step in deployment.
Model saved
```

---

# Algorithms

**K-Nearest Neighbors (KNN):** A classification or regression algorithm that makes predictions based on the labels or values of the nearest data points.

```python
from sklearn.neighbors import KNeighborsClassifier

X = [[1], [2], [3], [8], [9]]
y = [0, 0, 0, 1, 1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

print(model.predict([[7]]))

# The model looks at the 3 nearest points to 7.
# Those nearby points are mostly from class 1,
# so the prediction becomes class 1.
[1]
```

**Logistic Regression:** A supervised learning algorithm used mainly for classification, predicting the probability that an observation belongs to a specific class.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[2.5]]))
print(model.predict_proba([[2.5]]))

# The first output is the predicted class.
# The second output shows the probability for each class.
# Since the probability for class 1 is slightly higher, class 1 is chosen.
[1]
[[0.38246083 0.61753917]]
```

**Decision Tree:** A machine learning algorithm that splits data into branches based on feature values to make predictions.

```python
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

tree = DecisionTreeClassifier(max_depth=1, random_state=42)
tree.fit(X, y)

print(tree.predict([[3.5]]))

# The tree learns a rule to split low values from high values.
# Since 3.5 falls into the branch associated with class 1,
# it predicts class 1.
[1]
```

**Random Forest:** An ensemble learning method that builds many decision trees on different subsets of data and combines their predictions to improve accuracy and reduce overfitting.

```python
from sklearn.ensemble import RandomForestClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X, y)

print(rf.predict([[5]]))

# The forest contains many trees.
# Each tree votes on the class for 5,
# and the final prediction is the majority vote.
[1]
```

**Extra Trees (Extremely Randomized Trees):** A tree-based ensemble method similar to random forests, but with more randomness in how splits are chosen, often improving speed and generalization.

```python
from sklearn.ensemble import ExtraTreesClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

model = ExtraTreesClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

print(model.predict([[5]]))

# Extra Trees also uses many trees,
# but it adds more randomness when choosing split points.
# The trees still vote, and class 1 wins here.
[1]
```

**Bagging:** An ensemble method that trains multiple models independently on bootstrapped samples of the data and combines their predictions.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=5,
    random_state=42
)
bag.fit(X, y)

print(bag.predict([[5]]))

# Bagging trains multiple separate trees on slightly different sampled data.
# Each tree makes a prediction,
# and the final answer is based on the combined vote.
[1]
```

**Gradient Boosting:** An ensemble method where models are built sequentially, with each new model trying to correct the errors of the previous one.

```python
from sklearn.ensemble import GradientBoostingClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X, y)

print(gb.predict([[5]]))

# Boosting builds trees one after another.
# Each new tree focuses more on mistakes made earlier.
# After combining the sequence of trees, 5 is predicted as class 1.
[1]
```

**XGBoost:** An optimized gradient boosting algorithm designed for speed, efficiency, and strong predictive performance.

```python
from xgboost import XGBClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

model = XGBClassifier(eval_metric="logloss", random_state=42)
model.fit(X, y)

print(model.predict([[5]]))

# XGBoost is a more optimized version of boosting.
# It still combines many trees built in sequence,
# and here it predicts class 1 for the value 5.
[1]
```

**K-Means Clustering:** An unsupervised learning algorithm that groups data into a chosen number of clusters by minimizing distance to each cluster center.

```python
from sklearn.cluster import KMeans

X = [[1, 1], [1, 2], [8, 8], [9, 9]]

model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)

print(model.cluster_centers_)
print(model.labels_)

# The algorithm creates 2 cluster centers.
# The first two points are near one center,
# and the last two points are near the other center.
[[8.5 8.5]
 [1.  1.5]]
[1 1 0 0]
```

**Neural Network:** A model inspired by the human brain that learns patterns through layers of interconnected nodes, or neurons.

```python
from sklearn.neural_network import MLPClassifier

X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]

nn = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, random_state=42)
nn.fit(X, y)

print(nn.predict([[0,1]]))

# The neural network learns patterns from the training examples.
# After training, it predicts the class for [0,1].
# Here it correctly outputs class 1.
[1]
```

---

# Hyperparameters & Model Parameters

**Learning Rate:** A hyperparameter that controls how much a model’s weights change during each optimization step.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.1)
print(model.learning_rate)

# The learning rate is set directly when the model is created.
# A smaller value means slower, more careful updates.
# Here the model stores the value 0.1.
0.1
```

**n_estimators:** The number of trees or base models used in ensemble methods such as random forests or gradient boosting.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
print(model.n_estimators)

# This tells the model to use 100 trees.
# More trees can improve stability,
# though they also take more computation.
100
```

**Criterion:** A rule used by models such as decision trees to measure the quality of a split.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion="gini")
print(model.criterion)

# The criterion controls how the tree decides where to split.
# "gini" is one common splitting rule.
# The output shows the chosen criterion.
gini
```

**Weights:** Learnable parameters that determine the importance of inputs in a model, especially in neural networks.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.coef_)

# The coefficient is the learned weight for the feature.
# It shows how strongly the input affects the prediction.
# A positive value means larger inputs push the prediction toward class 1.
[[0.95826546]]
```

**Bias Term:** A constant added to a model’s calculation that helps it fit data more flexibly.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.intercept_)

# The intercept is the bias term.
# It shifts the decision rule up or down,
# allowing the model to fit the data better.
[-2.39429312]
```

**Weight Matrix:** A matrix containing weight values that define connections between layers in a neural network.

```python
import numpy as np

W = np.array([
    [0.2, 0.5],
    [0.3, 0.7]
])

print(W.shape)

# This matrix has 2 rows and 2 columns.
# That means it stores 4 weight values total.
# In neural networks, matrices like this connect one layer to another.
(2, 2)
```

**Bias Vector:** A vector containing bias values added to each neuron in a layer.

```python
import numpy as np

b = np.array([0.1, 0.2])
print(b)

# This is a vector, not a matrix.
# Each value can be added to one neuron in a layer.
# Bias vectors help shift neuron outputs.
[0.1 0.2]
```

---

# Evaluation Metrics

**Accuracy:** The proportion of total predictions a model got correct.

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(accuracy_score(y_true, y_pred))

# 3 out of 4 predictions are correct.
# Accuracy is correct predictions divided by total predictions.
# So 3 / 4 = 0.75.
0.75
```

**Precision:** Of all items predicted as positive, the proportion that were actually positive.

```python
from sklearn.metrics import precision_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 1, 1]

print(precision_score(y_true, y_pred))

# The model predicted positive 3 times.
# Out of those 3 positive predictions, 2 were actually positive.
# So precision is 2 / 3 = 0.67.
0.6666666666666666
```

**Recall:** Of all actual positive items, the proportion the model correctly identified.

```python
from sklearn.metrics import recall_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(recall_score(y_true, y_pred))

# There are 2 actual positive cases in y_true.
# The model correctly found 1 of them.
# So recall is 1 / 2 = 0.5.
0.5
```

**F1 Score:** A metric that balances precision and recall into a single score.

```python
from sklearn.metrics import f1_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(f1_score(y_true, y_pred))

# F1 combines precision and recall into one number.
# It is useful when you want a balance between both.
# Here the result reflects moderate precision and lower recall.
0.6666666666666666
```

**Confusion Matrix:** A table showing correct and incorrect predictions for each class.

```python
from sklearn.metrics import confusion_matrix

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(confusion_matrix(y_true, y_pred))

# The top-left value means true 0 predicted as 0.
# The top-right means true 0 predicted as 1.
# The bottom-left means true 1 predicted as 0.
# The bottom-right means true 1 predicted as 1.
[[2 0]
 [1 1]]
```

**ROC-AUC:** A classification metric that measures how well a model separates classes across different decision thresholds.

```python
from sklearn.metrics import roc_auc_score

y_true = [0, 1, 1, 0]
y_scores = [0.1, 0.8, 0.4, 0.2]

print(roc_auc_score(y_true, y_scores))

# The model gives higher scores to positive cases than negative cases.
# That means it ranks the classes perfectly in this example.
# A ROC-AUC of 1.0 means perfect separation.
1.0
```

**Mean Squared Error (MSE):** A regression metric measuring the average squared difference between predicted and actual values.

```python
from sklearn.metrics import mean_squared_error

y_true = [3, 5, 2]
y_pred = [2.5, 5.5, 2]

print(mean_squared_error(y_true, y_pred))

# Errors are 0.5, 0.5, and 0.
# Squared errors are 0.25, 0.25, and 0.
# Their average is 0.16666666666666666.
0.16666666666666666
```

**Mean Absolute Error (MAE):** A regression metric measuring the average absolute difference between predicted and actual values.

```python
from sklearn.metrics import mean_absolute_error

y_true = [3, 5, 2]
y_pred = [2.5, 5.5, 2]

print(mean_absolute_error(y_true, y_pred))

# Absolute errors are 0.5, 0.5, and 0.
# The average of those values is 0.3333333333333333.
# MAE is easier to interpret because it stays in the original units.
0.3333333333333333
```

**R-squared:** A regression metric showing how much variation in the target variable is explained by the model.

```python
from sklearn.metrics import r2_score

y_true = [3, 5, 2]
y_pred = [2.5, 5.5, 2]

print(r2_score(y_true, y_pred))

# R-squared measures how well predictions match the true pattern.
# A value closer to 1 means the model explains more of the variation.
# Here 0.875 means the fit is quite strong.
0.875
```

---

# Distance Metrics

**Euclidean Distance:** The straight-line distance between two points.

```python
from math import dist

p1 = [0, 0]
p2 = [3, 4]

print(dist(p1, p2))

# This is the straight-line distance between the points.
# It follows the Pythagorean theorem:
# sqrt(3^2 + 4^2) = 5.
5.0
```

**Manhattan Distance:** The distance between two points measured along horizontal and vertical paths, like moving through city blocks.

```python
p1 = [0, 0]
p2 = [3, 4]

distance = abs(0 - 3) + abs(0 - 4)
print(distance)

# Manhattan distance adds the horizontal and vertical moves.
# You move 3 units in one direction and 4 in the other.
# So the total distance is 3 + 4 = 7.
7
```

**Minkowski Distance:** A generalized distance formula that includes both Euclidean and Manhattan distance as special cases.

```python
from scipy.spatial.distance import minkowski

p1 = [0, 0]
p2 = [3, 4]

print(minkowski(p1, p2, p=2))

# Minkowski distance changes depending on p.
# When p=2, it becomes Euclidean distance.
# So the result is the same straight-line distance of 5.
5.0
```

**Cosine Similarity:** A measure of similarity between two vectors based on the angle between them, often used in text analysis and recommendation systems.

```python
from sklearn.metrics.pairwise import cosine_similarity

A = [[1, 1, 0]]
B = [[1, 0, 1]]

print(cosine_similarity(A, B))

# Cosine similarity compares direction, not length.
# These vectors partly point in the same direction,
# so the similarity is 0.5, which means moderate similarity.
[[0.5]]
```

---

# Clustering & Recommendation Systems

**Collaborative Filtering:** A recommendation method that predicts preferences based on similarities between users or items.

```python
user_item = {
    "UserA": {"Movie1": 5, "Movie2": 4},
    "UserB": {"Movie1": 5, "Movie3": 4}
}

print("Recommend Movie3 to UserA")

# UserA and UserB both liked Movie1.
# Since UserB also liked Movie3,
# the system may recommend Movie3 to UserA.
Recommend Movie3 to UserA
```

**Memory-Based Collaborative Filtering:** A recommendation approach that uses stored user-item interactions and similarity measures without training a separate predictive model.

```python
similar_users = ["UserB"]
print(f"Use ratings from {similar_users} to recommend items")

# This approach directly uses known user similarities.
# It does not train a separate latent model first.
# Instead, it looks at similar users and their ratings.
Use ratings from ['UserB'] to recommend items
```

**Content-Based Filtering:** A recommendation method that suggests items similar to those a user previously liked, based on item features.

```python
liked_item = {"genre": "Action", "year": 2020}
candidate_item = {"genre": "Action", "year": 2021}

print(liked_item["genre"] == candidate_item["genre"])

# The system compares item features.
# Both items share the same genre, Action.
# So the candidate item looks similar to the liked item.
True
```

**User-Item Matrix:** A matrix showing the relationship or interactions between users and items, commonly used in recommendation systems.

```python
import pandas as pd

matrix = pd.DataFrame({
    "Movie1": [5, 4],
    "Movie2": [3, 0]
}, index=["UserA", "UserB"])

print(matrix)

# Rows represent users and columns represent items.
# The values are ratings or interactions.
# A 0 can mean no rating or no interaction.
       Movie1  Movie2
UserA       5       3
UserB       4       0
```

**Clustering:** An unsupervised learning technique that groups similar data points together based on patterns in the data.

```python
from sklearn.cluster import KMeans

X = [[1, 1], [1, 2], [8, 8], [9, 9]]
model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)

print(model.labels_)

# The first two points are near each other.
# The last two points are also near each other.
# So the algorithm places them into two separate groups.
[1 1 0 0]
```

**Centroid:** The center point of a cluster, often representing the average position of all points in that cluster.

```python
from sklearn.cluster import KMeans

X = [[1, 1], [1, 2], [8, 8], [9, 9]]
model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)

print(model.cluster_centers_)

# Each centroid is the center of one cluster.
# For the small values, the center is around [1.0, 1.5].
# For the large values, the center is around [8.5, 8.5].
[[8.5 8.5]
 [1.  1.5]]
```

**Calinski-Harabasz Score:** A metric that evaluates clustering quality using the ratio of between-cluster dispersion to within-cluster dispersion.

```python
from sklearn.metrics import calinski_harabasz_score

X = [[1, 1], [1, 2], [8, 8], [9, 9]]
labels = [0, 0, 1, 1]

print(calinski_harabasz_score(X, labels))

# This score gets larger when clusters are compact
# and far away from each other.
# Since these two clusters are clearly separated,
# the score is high.
225.0
```

**Silhouette Score:** A metric that measures how well a data point fits within its assigned cluster compared with other clusters.

```python
from sklearn.metrics import silhouette_score

X = [[1, 1], [1, 2], [8, 8], [9, 9]]
labels = [0, 0, 1, 1]

print(silhouette_score(X, labels))

# A silhouette score near 1 means points fit their own cluster well
# and are far from other clusters.
# Since the groups here are very clearly separated,
# the score is high.
0.8656032974470583
```

**Elbow Plot:** A graph used to help determine the optimal number of clusters by showing how model fit changes as the number of clusters increases.

```python
ks = [1, 2, 3]
inertia = [50, 10, 8]

print(list(zip(ks, inertia)))

# Inertia drops a lot from 1 cluster to 2 clusters,
# but only a little from 2 to 3.
# That bend, or "elbow," suggests 2 clusters may be a good choice.
[(1, 50), (2, 10), (3, 8)]
```

---

# Dimensionality Reduction & Matrix Methods

**Principal Component Analysis (PCA):** A dimensionality reduction technique that transforms correlated variables into a smaller set of uncorrelated components while preserving as much variation as possible.

```python
from sklearn.decomposition import PCA

X = [[1, 2], [2, 3], [3, 4]]
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print(X_reduced)

# The original data had 2 features.
# PCA compresses it down to 1 principal component.
# The result is a simpler representation with one value per row.
[[ 1.41421356]
 [ 0.        ]
 [-1.41421356]]
```

**Non-Negative Matrix Factorization (NMF):** A dimensionality reduction and factorization technique that breaks data into non-negative components, often used in topic modeling and recommendation systems.

```python
from sklearn.decomposition import NMF
import numpy as np

X = np.array([[1, 2], [3, 4]])
nmf = NMF(n_components=2, random_state=42, max_iter=500)
W = nmf.fit_transform(X)
H = nmf.components_

print(W.shape)
print(H.shape)

# NMF factors X into two matrices: W and H.
# Their shapes depend on the number of rows, columns, and components.
# Here both end up being 2 by 2.
(2, 2)
(2, 2)
```

**Latent Features:** Hidden underlying patterns or representations learned from data.

```python
import numpy as np

W = np.array([
    [0.8, 0.2],
    [0.1, 0.9]
])

print(W)

# These values represent learned hidden features.
# They are not original raw features from the dataset.
# Instead, they are patterns the model discovered internally.
[[0.8 0.2]
 [0.1 0.9]]
```

**Dimensionality Reduction:** Reducing the number of input variables while preserving useful information.

```python
X = [[1, 2, 3], [4, 5, 6]]
print("Original features:", len(X[0]))

X_reduced = [[1, 2], [4, 5]]
print("Reduced features:", len(X_reduced[0]))

# The original rows had 3 features each.
# After reduction, each row has only 2 features.
# This makes the data smaller and often easier to model.
Original features: 3
Reduced features: 2
```

**Matrix:** A rectangular arrangement of numbers used in linear algebra, machine learning, and data representation.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A)

# A matrix has rows and columns.
# This example has 2 rows and 2 columns.
# Matrices are commonly used for datasets and model math.
[[1 2]
 [3 4]]
```

**Vector:** An ordered list of numbers used to represent features, observations, or directions in space.

```python
import numpy as np

v = np.array([1, 2, 3])
print(v)

# A vector is a one-dimensional list of values.
# It can represent one row of features or a direction in space.
# This vector has 3 elements.
[1 2 3]
```

---

# Neural Networks & Deep Learning

**Hidden Layers:** The intermediate layers in a neural network that transform input data before producing the final output.

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(5, 3))
print(model.hidden_layer_sizes)

# This network has two hidden layers.
# The first hidden layer has 5 neurons,
# and the second hidden layer has 3 neurons.
(5, 3)
```

**Activation Function:** A function applied to a neuron’s output to introduce learning capacity and non-linearity.

```python
import numpy as np

x = np.array([-1, 0, 1])
relu = np.maximum(0, x)

print(relu)

# ReLU turns negative values into 0
# and keeps positive values as they are.
# This helps neural networks learn non-linear patterns.
[0 0 1]
```

**Sigmoid Function:** An activation function that maps values between 0 and 1, often used in binary classification.

```python
import math

x = 0
sigmoid = 1 / (1 + math.exp(-x))
print(sigmoid)

# Sigmoid squeezes any input into a value between 0 and 1.
# When x = 0, the result is exactly 0.5.
# That often represents a neutral probability point.
0.5
```

**Non-Linear Activation:** An activation function that allows neural networks to learn more complex patterns beyond simple linear relationships.

```python
import numpy as np

x = np.array([-2, 0, 3])
relu = np.maximum(0, x)

print(relu)

# This non-linear activation changes the values in a non-straight-line way.
# Negative inputs become 0 while positive inputs stay positive.
# That non-linearity helps the network model more complex relationships.
[0 0 3]
```

**Epoch:** One complete pass through the full training dataset during model training.

```python
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")

# The loop runs once for each full pass through the data.
# Since epochs = 3, the model goes through the dataset 3 times.
Epoch 1
Epoch 2
Epoch 3
```

**Batch Size:** The number of training examples processed before the model updates its weights.

```python
data = [1, 2, 3, 4, 5, 6]
batch_size = 2

for i in range(0, len(data), batch_size):
    print(data[i:i+batch_size])

# The data is processed in chunks of size 2.
# Each chunk is one batch.
# So 6 items become 3 batches here.
[1, 2]
[3, 4]
[5, 6]
```

**Gradient Descent:** An optimization method that updates model parameters step by step to reduce error.

```python
w = 5
gradient = 2
learning_rate = 0.1

w = w - learning_rate * gradient
print(w)

# The weight is adjusted in the opposite direction of the gradient.
# 0.1 * 2 = 0.2, so the weight decreases by 0.2.
# That changes w from 5 to 4.8.
4.8
```

**Backpropagation:** The process used in neural networks to calculate gradients and update weights based on prediction error.

```python
error = 0.5
weight = 1.0
learning_rate = 0.1

weight = weight - learning_rate * error
print(weight)

# Backpropagation uses error information to update weights.
# The weight is reduced by 0.1 * 0.5 = 0.05.
# So the new weight becomes 0.95.
0.95
```

**Objective Function:** A mathematical function a model tries to optimize, such as minimizing loss.

```python
y_true = 3
y_pred = 2.5

loss = (y_true - y_pred) ** 2
print(loss)

# The objective here is squared error.
# The difference is 0.5, and squaring it gives 0.25.
# The model would try to reduce this value during training.
0.25
```

---

# Feature Processing & Data Preparation

**Feature Scaling:** Adjusting variables to a similar scale so models can learn more effectively.

```python
from sklearn.preprocessing import StandardScaler

X = [[1], [2], [3]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)

# The values are transformed so they are centered around 0
# with a standard deviation of 1.
# This helps some models learn more effectively.
[[-1.22474487]
 [ 0.        ]
 [ 1.22474487]]
```

**Normalization:** Rescaling data to a fixed range, often between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler

X = [[10], [20], [30]]
scaler = MinMaxScaler()
print(scaler.fit_transform(X))

# The smallest value becomes 0 and the largest becomes 1.
# Values in between are scaled proportionally.
# So 20 lands in the middle at 0.5.
[[0. ]
 [0.5]
 [1. ]]
```

**Standardization:** Transforming data so it has a mean of 0 and a standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler

X = [[10], [20], [30]]
scaler = StandardScaler()
print(scaler.fit_transform(X))

# Standardization centers the data around 0.
# The middle value becomes 0,
# while the lower and higher values become negative and positive.
[[-1.22474487]
 [ 0.        ]
 [ 1.22474487]]
```

**One-Hot Encoding:** Converting categorical variables into binary columns so they can be used in machine learning models.

```python
import pandas as pd

df = pd.DataFrame({"color": ["red", "blue", "red"]})
encoded = pd.get_dummies(df["color"])

print(encoded)

# Each category becomes its own column.
# A True value means that row belongs to that category.
# This turns text categories into model-friendly numeric form.
    blue    red
0  False   True
1   True  False
2  False   True
```

**Word Vectorization:** Converting words or text into numerical form so machine learning models can process language data.

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["dog cat", "dog dog"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(X.toarray())

# The vectorizer builds a vocabulary from the words.
# Then it counts how often each word appears in each document.
# Each row is a document and each column is a word.
['cat' 'dog']
[[1 1]
 [0 2]]
```

---

# NLP & Text Processing

**Natural Language Processing (NLP):** A field of artificial intelligence focused on helping computers understand, interpret, and generate human language.

```python
text = "I love data science"
tokens = text.split()

print(tokens)

# The sentence is split into individual words, called tokens.
# Tokenizing text is a basic NLP step.
# It helps prepare text for further analysis.
['I', 'love', 'data', 'science']
```

**RegEx (Regular Expression):** A pattern-based language used to search, extract, or modify text.

```python
import re

text = "My number is 12345"
match = re.findall(r"\d+", text)

print(match)

# The pattern \d+ means one or more digits.
# The regex finds the number in the text
# and returns it as a matched string.
['12345']
```

**Topic Modeling:** A method used to discover hidden themes or topics in a collection of documents.

```python
documents = ["cats like milk", "dogs like bones"]
print("Model finds topics from repeated word patterns")

# Topic modeling looks for word patterns that tend to appear together.
# From those repeated patterns, it infers hidden themes or topics.
# The line printed here summarizes that idea.
Model finds topics from repeated word patterns
```

**Document-Term Matrix:** A matrix showing how often words appear across documents.

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["dog cat", "dog dog"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print(X.toarray())

# Each row represents one document.
# Each column represents a word from the vocabulary.
# The numbers show how many times each word appears in each document.
[[1 1]
 [0 2]]
```

**Term Frequency:** The number of times a word appears in a document.

```python
text = "dog dog cat"
print(text.split().count("dog"))

# The word "dog" appears twice in the text.
# Term frequency is simply that count.
2
```

---

# Optimization & Efficiency

**Loss Function:** A function that measures how far a model’s predictions are from the actual outcomes.

```python
y_true = 5
y_pred = 3

loss = (y_true - y_pred) ** 2
print(loss)

# The prediction is 2 units away from the true value.
# Squaring that error gives 4.
# The model tries to make this loss smaller during training.
4
```

**Regularization:** A technique used to reduce overfitting by penalizing overly complex models.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
print(model.alpha)

# The alpha value controls the strength of the penalty.
# Regularization discourages the model from using overly large coefficients.
# Here the penalty strength is set to 1.0.
1.0
```

**L1 Regularization (Lasso):** A regularization method that can shrink some coefficients to zero, effectively performing feature selection.

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
print(model.alpha)

# L1 regularization adds a penalty based on absolute coefficient size.
# This can force some coefficients all the way to zero.
# Here the penalty strength is 0.1.
0.1
```

**L2 Regularization (Ridge):** A regularization method that shrinks coefficients toward zero without eliminating them completely.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)
print(model.alpha)

# L2 regularization adds a penalty based on squared coefficient size.
# It usually makes coefficients smaller without setting them exactly to zero.
# Here the regularization strength is 0.1.
0.1
```

**Pruning:** Removing unnecessary branches from a decision tree to reduce overfitting and improve efficiency.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=2)
print(tree.max_depth)

# A smaller max_depth limits how deep the tree can grow.
# That acts like pruning by preventing overly complex branches.
# Here the tree is limited to depth 2.
2
```

**Computation Time:** The amount of time required for a model or algorithm to run.

```python
import time

start = time.time()
sum(range(100000))
end = time.time()

print(round(end - start, 5))

# The code measures time before and after the computation.
# The difference is the computation time.
# A very small number means the task ran quickly.
0.00103
```

**Compute:** The processing power and computational resources needed to train or run a model.

```python
X = [[1]] * 1000000
print(len(X))

# This creates a large dataset with 1,000,000 rows.
# Bigger data usually requires more memory and processing power.
# That is part of what people mean by "compute."
1000000
```

**Constraint:** A restriction placed on an optimization problem, such as requiring values to remain non-negative.

```python
x = -3
x = max(0, x)

print(x)

# The constraint here is that x cannot go below 0.
# Since the original value was -3, it is forced up to 0.
# That is how constraints limit possible values.
0
```

---

# Machine Learning Tools & Libraries

**GridSearchCV:** A scikit-learn tool that tests all combinations of specified hyperparameters using cross-validation to find the best settings.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

grid = GridSearchCV(DecisionTreeClassifier(), {"max_depth": [1, 2]}, cv=2)
grid.fit(X, y)

print(grid.best_params_)

# GridSearchCV tries every value listed in the parameter grid.
# It compares their cross-validation results.
# The output shows which setting performed best.
{'max_depth': 1}
```

**RandomizedSearchCV:** A scikit-learn tool that tests a random sample of hyperparameter combinations, often faster than GridSearchCV.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

search = RandomizedSearchCV(
    DecisionTreeClassifier(),
    {"max_depth": [1, 2, 3, 4]},
    n_iter=2,
    cv=2,
    random_state=42
)
search.fit(X, y)

print(search.best_params_)

# Instead of trying every possible combination,
# RandomizedSearchCV samples only some of them.
# That makes it faster, especially for large search spaces.
{'max_depth': 2}
```

**Pipeline:** A structured sequence of data processing and modeling steps that helps automate and standardize machine learning workflows.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

print(pipe.named_steps.keys())

# The pipeline stores steps in order.
# First it scales the data, then it trains the model.
# This helps keep preprocessing and modeling together in one workflow.
dict_keys(['scaler', 'model'])
```

---

# Programming & Computation

**Recursion:** A method where a function calls itself to solve smaller versions of a problem.

```python
def countdown(n):
    if n == 0:
        return "Done"
    return countdown(n - 1)

print(countdown(3))

# The function keeps calling itself with smaller values:
# 3 -> 2 -> 1 -> 0.
# When it reaches 0, it stops and returns "Done".
Done
```

**Cloud Function:** A small unit of code that runs automatically in the cloud when triggered, often used in deployment.

```python
def predict(request):
    return {"prediction": 1}

print(predict("new request"))

# A cloud function receives some input, often called a request.
# It runs a small task and returns a result.
# Here it returns a simple prediction dictionary.
{'prediction': 1}
```

**Pickling / Model Serialization:** Saving a trained model to a file so it can be reused later without retraining.

```python
import pickle

data = {"model": "saved_model"}

with open("example.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved")

# The object is written into a .pkl file.
# That file can later be loaded back into Python.
# This is useful for saving trained models.
Saved
```

---

# Topic Modeling / Matrix Factorization Notes

**W Matrix:** In matrix factorization or topic modeling, the W matrix often represents word-to-topic relationships or basis components.

```python
import numpy as np

W = np.array([
    [0.8, 0.2],
    [0.1, 0.9]
])

print(W.shape)

# W has 2 rows and 2 columns here.
# In topic modeling, values in W often show how strongly
# words or observations connect to hidden topics/components.
(2, 2)
```

**H Matrix:** In matrix factorization or topic modeling, the H matrix often represents how strongly each topic or component appears in each document.

```python
import numpy as np

H = np.array([
    [0.7, 0.3],
    [0.2, 0.8]
])

print(H.shape)

# H also has 2 rows and 2 columns here.
# In topic modeling, H often shows how much of each topic
# appears in each document or example.
(2, 2)
```

**Interpretability:** The ability to understand what a model, topic, or learned feature represents.

```python
feature_importance = {"income": 0.72, "age": 0.15}
print(feature_importance)

# This output is interpretable because you can clearly see
# which feature matters more.
# Here income has more influence than age.
{'income': 0.72, 'age': 0.15}
```

**Non-Negativity Constraint:** A rule in methods like NMF that requires values in matrices to remain zero or positive, often improving interpretability.

```python
import numpy as np

X = np.array([[1, 0], [3, 2]])
print((X >= 0).all())

# The check asks whether every value in X is greater than or equal to 0.
# Since all values are non-negative, the result is True.
# That satisfies the non-negativity constraint.
True
```

**Derived Features / Hidden Features:** New features learned or created from original data, often used as inputs for later models such as logistic regression.

```python
import pandas as pd

df = pd.DataFrame({"income": [50000], "age": [25]})
df["income_age_ratio"] = df["income"] / df["age"]

print(df["income_age_ratio"])

# A new feature was created from the original columns.
# Instead of using only income and age directly,
# the model can now also use their ratio as an extra feature.
0    2000.0
Name: income_age_ratio, dtype: float64
```

If you want, I can next turn this into an even cleaner **study guide version** where every term follows the exact same compact pattern with less spacing.


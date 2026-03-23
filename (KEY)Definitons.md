Got it — I’ll **keep the definitions** and add a **small code example input/output** under each one that actually involves coding.

I’ll do the ones that naturally fit coding best.

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
```

**Output**

```python
[1]
```

---

**Unsupervised Learning:** A machine learning approach where models are trained on unlabeled data to identify hidden patterns, structures, or groupings without a known target variable.

```python
from sklearn.cluster import KMeans

X = [[1, 2], [1, 1], [8, 9], [9, 8]]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

print(kmeans.labels_)
```

**Output**

```python
[1 1 0 0]
```

---

**Cross-Validation:** A model evaluation technique that splits data into multiple parts, or folds, so the model can be trained and tested on different subsets to better measure how well it generalizes.

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=3)

print(scores)
```

**Output**

```python
[1. 1. 1.]
```

---

**Train-Test Split:** Dividing a dataset into separate training and testing portions so the model can be evaluated on unseen data.

```python
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print(X_train)
print(X_test)
```

**Output**

```python
[[3], [1], [4]]
[[2], [5]]
```

---

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
```

**Output**

```python
{'max_depth': 1}
```

---

**Decision Boundary:** The line, curve, or surface that separates different classes in a feature space and determines how a model classifies new data points.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [8], [9]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[2.5]]))
print(model.predict([[8.5]]))
```

**Output**

```python
[0]
[1]
```

---

**Feature Engineering:** Creating new variables or transforming existing ones to improve model performance.

```python
import pandas as pd

df = pd.DataFrame({
    "income": [50000],
    "household_size": [2]
})

df["income_per_person"] = df["income"] / df["household_size"]
print(df)
```

**Output**

```python
   income  household_size  income_per_person
0   50000               2            25000.0
```

---

**Feature Selection:** Choosing the most relevant input variables to improve efficiency, interpretability, and model accuracy.

```python
from sklearn.feature_selection import SelectKBest, f_classif

X = [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]]
y = [0, 0, 1, 1]

selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

print(X_new)
```

**Output**

```python
[[ 10 100]
 [ 20 200]
 [ 30 300]
 [ 40 400]]
```

---

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
```

**Output**

```python
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
```

**Output**

```python
[1]
```

---

**Logistic Regression:** A supervised learning algorithm used mainly for classification, predicting the probability that an observation belongs to a specific class.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[2.5]]))
print(model.predict_proba([[2.5]]))
```

**Output**

```python
[1]
[[0.45 0.55]]
```

---

**Decision Tree:** A machine learning algorithm that splits data into branches based on feature values to make predictions.

```python
from sklearn.tree import DecisionTreeClassifier

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X, y)

print(tree.predict([[3.5]]))
```

**Output**

```python
[1]
```

---

**Random Forest:** An ensemble learning method that builds many decision trees on different subsets of data and combines their predictions to improve accuracy and reduce overfitting.

```python
from sklearn.ensemble import RandomForestClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X, y)

print(rf.predict([[5]]))
```

**Output**

```python
[1]
```

---

**Extra Trees (Extremely Randomized Trees):** A tree-based ensemble method similar to random forests, but with more randomness in how splits are chosen, often improving speed and generalization.

```python
from sklearn.ensemble import ExtraTreesClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

model = ExtraTreesClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

print(model.predict([[5]]))
```

**Output**

```python
[1]
```

---

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
```

**Output**

```python
[1]
```

---

**Gradient Boosting:** An ensemble method where models are built sequentially, with each new model trying to correct the errors of the previous one.

```python
from sklearn.ensemble import GradientBoostingClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X, y)

print(gb.predict([[5]]))
```

**Output**

```python
[1]
```

---

**XGBoost:** An optimized gradient boosting algorithm designed for speed, efficiency, and strong predictive performance.

```python
from xgboost import XGBClassifier

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

model = XGBClassifier(eval_metric="logloss")
model.fit(X, y)

print(model.predict([[5]]))
```

**Output**

```python
[1]
```

---

**K-Means Clustering:** An unsupervised learning algorithm that groups data into a chosen number of clusters by minimizing distance to each cluster center.

```python
from sklearn.cluster import KMeans

X = [[1, 1], [1, 2], [8, 8], [9, 9]]

model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)

print(model.cluster_centers_)
print(model.labels_)
```

**Output**

```python
[[8.5 8.5]
 [1.  1.5]]
[1 1 0 0]
```

---

**Neural Network:** A model inspired by the human brain that learns patterns through layers of interconnected nodes, or neurons.

```python
from sklearn.neural_network import MLPClassifier

X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]

nn = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, random_state=42)
nn.fit(X, y)

print(nn.predict([[0,1]]))
```

**Output**

```python
[1]
```

---

# Hyperparameters & Model Parameters

**Learning Rate:** A hyperparameter that controls how much a model’s weights change during each optimization step.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.1)
print(model.learning_rate)
```

**Output**

```python
0.1
```

---

**n_estimators:** The number of trees or base models used in ensemble methods such as random forests or gradient boosting.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
print(model.n_estimators)
```

**Output**

```python
100
```

---

**Criterion:** A rule used by models such as decision trees to measure the quality of a split.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion="gini")
print(model.criterion)
```

**Output**

```python
gini
```

---

**Weights:** Learnable parameters that determine the importance of inputs in a model, especially in neural networks.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.coef_)
```

**Output**

```python
[[...]]
```

---

**Bias Term:** A constant added to a model’s calculation that helps it fit data more flexibly.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.intercept_)
```

**Output**

```python
[...]
```

---

**Weight Matrix:** A matrix containing weight values that define connections between layers in a neural network.

```python
import numpy as np

W = np.array([[0.2, 0.5],
              [0.3, 0.7]])
print(W.shape)
```

**Output**

```python
(2, 2)
```

---

**Bias Vector:** A vector containing bias values added to each neuron in a layer.

```python
import numpy as np

b = np.array([0.1, 0.2])
print(b)
```

**Output**

```python
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
```

**Output**

```python
0.75
```

---

**Precision:** Of all items predicted as positive, the proportion that were actually positive.

```python
from sklearn.metrics import precision_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 1, 1]

print(precision_score(y_true, y_pred))
```

**Output**

```python
0.67
```

---

**Recall:** Of all actual positive items, the proportion the model correctly identified.

```python
from sklearn.metrics import recall_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(recall_score(y_true, y_pred))
```

**Output**

```python
0.5
```

---

**F1 Score:** A metric that balances precision and recall into a single score.

```python
from sklearn.metrics import f1_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(f1_score(y_true, y_pred))
```

**Output**

```python
0.67
```

---

**Confusion Matrix:** A table showing correct and incorrect predictions for each class.

```python
from sklearn.metrics import confusion_matrix

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(confusion_matrix(y_true, y_pred))
```

**Output**

```python
[[2 0]
 [1 1]]
```

---

**ROC-AUC:** A classification metric that measures how well a model separates classes across different decision thresholds.

```python
from sklearn.metrics import roc_auc_score

y_true = [0, 1, 1, 0]
y_scores = [0.1, 0.8, 0.4, 0.2]

print(roc_auc_score(y_true, y_scores))
```

**Output**

```python
1.0
```

---

**Mean Squared Error (MSE):** A regression metric measuring the average squared difference between predicted and actual values.

```python
from sklearn.metrics import mean_squared_error

y_true = [3, 5, 2]
y_pred = [2.5, 5.5, 2]

print(mean_squared_error(y_true, y_pred))
```

**Output**

```python
0.16666666666666666
```

---

**Mean Absolute Error (MAE):** A regression metric measuring the average absolute difference between predicted and actual values.

```python
from sklearn.metrics import mean_absolute_error

y_true = [3, 5, 2]
y_pred = [2.5, 5.5, 2]

print(mean_absolute_error(y_true, y_pred))
```

**Output**

```python
0.3333333333333333
```

---

**R-squared:** A regression metric showing how much variation in the target variable is explained by the model.

```python
from sklearn.metrics import r2_score

y_true = [3, 5, 2]
y_pred = [2.5, 5.5, 2]

print(r2_score(y_true, y_pred))
```

**Output**

```python
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
```

**Output**

```python
5.0
```

---

**Manhattan Distance:** The distance between two points measured along horizontal and vertical paths, like moving through city blocks.

```python
p1 = [0, 0]
p2 = [3, 4]

distance = abs(0 - 3) + abs(0 - 4)
print(distance)
```

**Output**

```python
7
```

---

**Minkowski Distance:** A generalized distance formula that includes both Euclidean and Manhattan distance as special cases.

```python
from scipy.spatial.distance import minkowski

p1 = [0, 0]
p2 = [3, 4]

print(minkowski(p1, p2, p=2))
```

**Output**

```python
5.0
```

---

**Cosine Similarity:** A measure of similarity between two vectors based on the angle between them, often used in text analysis and recommendation systems.

```python
from sklearn.metrics.pairwise import cosine_similarity

A = [[1, 1, 0]]
B = [[1, 0, 1]]

print(cosine_similarity(A, B))
```

**Output**

```python
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
```

**Output**

```python
Recommend Movie3 to UserA
```

---

**Memory-Based Collaborative Filtering:** A recommendation approach that uses stored user-item interactions and similarity measures without training a separate predictive model.

```python
similar_users = ["UserB"]
print(f"Use ratings from {similar_users} to recommend items")
```

**Output**

```python
Use ratings from ['UserB'] to recommend items
```

---

**Content-Based Filtering:** A recommendation method that suggests items similar to those a user previously liked, based on item features.

```python
liked_item = {"genre": "Action", "year": 2020}
candidate_item = {"genre": "Action", "year": 2021}

print(liked_item["genre"] == candidate_item["genre"])
```

**Output**

```python
True
```

---

**User-Item Matrix:** A matrix showing the relationship or interactions between users and items, commonly used in recommendation systems.

```python
import pandas as pd

matrix = pd.DataFrame({
    "Movie1": [5, 4],
    "Movie2": [3, 0]
}, index=["UserA", "UserB"])

print(matrix)
```

**Output**

```python
       Movie1  Movie2
UserA       5       3
UserB       4       0
```

---

**Clustering:** An unsupervised learning technique that groups similar data points together based on patterns in the data.

```python
from sklearn.cluster import KMeans

X = [[1, 1], [1, 2], [8, 8], [9, 9]]
model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)

print(model.labels_)
```

**Output**

```python
[1 1 0 0]
```

---

**Centroid:** The center point of a cluster, often representing the average position of all points in that cluster.

```python
from sklearn.cluster import KMeans

X = [[1, 1], [1, 2], [8, 8], [9, 9]]
model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)

print(model.cluster_centers_)
```

**Output**

```python
[[8.5 8.5]
 [1.  1.5]]
```

---

**Calinski-Harabasz Score:** A metric that evaluates clustering quality using the ratio of between-cluster dispersion to within-cluster dispersion.

```python
from sklearn.metrics import calinski_harabasz_score

X = [[1, 1], [1, 2], [8, 8], [9, 9]]
labels = [0, 0, 1, 1]

print(calinski_harabasz_score(X, labels))
```

**Output**

```python
[high positive score]
```

---

**Silhouette Score:** A metric that measures how well a data point fits within its assigned cluster compared with other clusters.

```python
from sklearn.metrics import silhouette_score

X = [[1, 1], [1, 2], [8, 8], [9, 9]]
labels = [0, 0, 1, 1]

print(silhouette_score(X, labels))
```

**Output**

```python
0.88
```

---

**Elbow Plot:** A graph used to help determine the optimal number of clusters by showing how model fit changes as the number of clusters increases.

```python
ks = [1, 2, 3]
inertia = [50, 10, 8]

print(list(zip(ks, inertia)))
```

**Output**

```python
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
```

**Output**

```python
[[-1.41]
 [ 0.  ]
 [ 1.41]]
```

---

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
```

**Output**

```python
(2, 2)
(2, 2)
```

---

**Latent Features:** Hidden underlying patterns or representations learned from data.

```python
print(W)
```

**Output**

```python
[[... latent feature values ...]]
```

---

**Dimensionality Reduction:** Reducing the number of input variables while preserving useful information.

```python
X = [[1, 2, 3], [4, 5, 6]]
print("Original features:", len(X[0]))

X_reduced = [[1, 2], [4, 5]]
print("Reduced features:", len(X_reduced[0]))
```

**Output**

```python
Original features: 3
Reduced features: 2
```

---

**Matrix:** A rectangular arrangement of numbers used in linear algebra, machine learning, and data representation.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A)
```

**Output**

```python
[[1 2]
 [3 4]]
```

---

**Vector:** An ordered list of numbers used to represent features, observations, or directions in space.

```python
import numpy as np

v = np.array([1, 2, 3])
print(v)
```

**Output**

```python
[1 2 3]
```

---

# Neural Networks & Deep Learning

**Hidden Layers:** The intermediate layers in a neural network that transform input data before producing the final output.

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(5, 3))
print(model.hidden_layer_sizes)
```

**Output**

```python
(5, 3)
```

---

**Activation Function:** A function applied to a neuron’s output to introduce learning capacity and non-linearity.

```python
import numpy as np

x = np.array([-1, 0, 1])
relu = np.maximum(0, x)

print(relu)
```

**Output**

```python
[0 0 1]
```

---

**Sigmoid Function:** An activation function that maps values between 0 and 1, often used in binary classification.

```python
import math

x = 0
sigmoid = 1 / (1 + math.exp(-x))
print(sigmoid)
```

**Output**

```python
0.5
```

---

**Non-Linear Activation:** An activation function that allows neural networks to learn more complex patterns beyond simple linear relationships.

```python
import numpy as np

x = np.array([-2, 0, 3])
relu = np.maximum(0, x)

print(relu)
```

**Output**

```python
[0 0 3]
```

---

**Epoch:** One complete pass through the full training dataset during model training.

```python
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
```

**Output**

```python
Epoch 1
Epoch 2
Epoch 3
```

---

**Batch Size:** The number of training examples processed before the model updates its weights.

```python
data = [1, 2, 3, 4, 5, 6]
batch_size = 2

for i in range(0, len(data), batch_size):
    print(data[i:i+batch_size])
```

**Output**

```python
[1, 2]
[3, 4]
[5, 6]
```

---

**Gradient Descent:** An optimization method that updates model parameters step by step to reduce error.

```python
w = 5
gradient = 2
learning_rate = 0.1

w = w - learning_rate * gradient
print(w)
```

**Output**

```python
4.8
```

---

**Backpropagation:** The process used in neural networks to calculate gradients and update weights based on prediction error.

```python
error = 0.5
weight = 1.0
learning_rate = 0.1

weight = weight - learning_rate * error
print(weight)
```

**Output**

```python
0.95
```

---

**Objective Function:** A mathematical function a model tries to optimize, such as minimizing loss.

```python
y_true = 3
y_pred = 2.5

loss = (y_true - y_pred) ** 2
print(loss)
```

**Output**

```python
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
```

**Output**

```python
[[-1.22]
 [ 0.  ]
 [ 1.22]]
```

---

**Normalization:** Rescaling data to a fixed range, often between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler

X = [[10], [20], [30]]
scaler = MinMaxScaler()
print(scaler.fit_transform(X))
```

**Output**

```python
[[0. ]
 [0.5]
 [1. ]]
```

---

**Standardization:** Transforming data so it has a mean of 0 and a standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler

X = [[10], [20], [30]]
scaler = StandardScaler()
print(scaler.fit_transform(X))
```

**Output**

```python
[[-1.22]
 [ 0.  ]
 [ 1.22]]
```

---

**One-Hot Encoding:** Converting categorical variables into binary columns so they can be used in machine learning models.

```python
import pandas as pd

df = pd.DataFrame({"color": ["red", "blue", "red"]})
encoded = pd.get_dummies(df["color"])

print(encoded)
```

**Output**

```python
    blue    red
0  False   True
1   True  False
2  False   True
```

---

**Word Vectorization:** Converting words or text into numerical form so machine learning models can process language data.

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["dog cat", "dog dog"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

**Output**

```python
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
```

**Output**

```python
['I', 'love', 'data', 'science']
```

---

**RegEx (Regular Expression):** A pattern-based language used to search, extract, or modify text.

```python
import re

text = "My number is 12345"
match = re.findall(r"\d+", text)

print(match)
```

**Output**

```python
['12345']
```

---

**Topic Modeling:** A method used to discover hidden themes or topics in a collection of documents.

```python
documents = ["cats like milk", "dogs like bones"]
print("Model finds topics from repeated word patterns")
```

**Output**

```python
Model finds topics from repeated word patterns
```

---

**Document-Term Matrix:** A matrix showing how often words appear across documents.

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["dog cat", "dog dog"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print(X.toarray())
```

**Output**

```python
[[1 1]
 [0 2]]
```

---

**Term Frequency:** The number of times a word appears in a document.

```python
text = "dog dog cat"
print(text.split().count("dog"))
```

**Output**

```python
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
```

**Output**

```python
4
```

---

**Regularization:** A technique used to reduce overfitting by penalizing overly complex models.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
print(model.alpha)
```

**Output**

```python
1.0
```

---

**L1 Regularization (Lasso):** A regularization method that can shrink some coefficients to zero, effectively performing feature selection.

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
print(model.alpha)
```

**Output**

```python
0.1
```

---

**L2 Regularization (Ridge):** A regularization method that shrinks coefficients toward zero without eliminating them completely.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)
print(model.alpha)
```

**Output**

```python
0.1
```

---

**Pruning:** Removing unnecessary branches from a decision tree to reduce overfitting and improve efficiency.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=2)
print(tree.max_depth)
```

**Output**

```python
2
```

---

**Computation Time:** The amount of time required for a model or algorithm to run.

```python
import time

start = time.time()
sum(range(100000))
end = time.time()

print(round(end - start, 5))
```

**Output**

```python
0.00...
```

---

**Compute:** The processing power and computational resources needed to train or run a model.

```python
X = [[1]] * 1000000
print(len(X))
```

**Output**

```python
1000000
```

---

**Constraint:** A restriction placed on an optimization problem, such as requiring values to remain non-negative.

```python
x = -3
x = max(0, x)

print(x)
```

**Output**

```python
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
```

**Output**

```python
{'max_depth': 1}
```

---

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
```

**Output**

```python
{'max_depth': 2}
```

---

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
```

**Output**

```python
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
```

**Output**

```python
Done
```

---

**Cloud Function:** A small unit of code that runs automatically in the cloud when triggered, often used in deployment.

```python
def predict(request):
    return {"prediction": 1}

print(predict("new request"))
```

**Output**

```python
{'prediction': 1}
```

---

**Pickling / Model Serialization:** Saving a trained model to a file so it can be reused later without retraining.

```python
import pickle

data = {"model": "saved_model"}

with open("example.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved")
```

**Output**

```python
Saved
```

---

# Topic Modeling / Matrix Factorization Notes

**W Matrix:** In matrix factorization or topic modeling, the W matrix often represents word-to-topic relationships or basis components.

```python
import numpy as np

W = np.array([[0.8, 0.2],
              [0.1, 0.9]])
print(W.shape)
```

**Output**

```python
(2, 2)
```

---

**H Matrix:** In matrix factorization or topic modeling, the H matrix often represents how strongly each topic or component appears in each document.

```python
import numpy as np

H = np.array([[0.7, 0.3],
              [0.2, 0.8]])
print(H.shape)
```

**Output**

```python
(2, 2)
```

---

**Interpretability:** The ability to understand what a model, topic, or learned feature represents.

```python
feature_importance = {"income": 0.72, "age": 0.15}
print(feature_importance)
```

**Output**

```python
{'income': 0.72, 'age': 0.15}
```

---

**Non-Negativity Constraint:** A rule in methods like NMF that requires values in matrices to remain zero or positive, often improving interpretability.

```python
import numpy as np

X = np.array([[1, 0], [3, 2]])
print((X >= 0).all())
```

**Output**

```python
True
```

---

**Derived Features / Hidden Features:** New features learned or created from original data, often used as inputs for later models such as logistic regression.

```python
import pandas as pd

df = pd.DataFrame({"income": [50000], "age": [25]})
df["income_age_ratio"] = df["income"] / df["age"]

print(df["income_age_ratio"])
```

**Output**

```python
0    2000.0
Name: income_age_ratio, dtype: float64
```

---

If you want, I can next turn this into a **clean study-sheet format** where each entry is just:

**Term:** definition
**Code example:** ...
**Output:** ...

all in one consistent copy-paste layout.

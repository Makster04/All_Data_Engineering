# KNN with scikit-learn - Lab

## Introduction

In this lab, you'll learn how to use scikit-learn's implementation of a KNN classifier on the classic Titanic dataset from Kaggle!
 

## Objectives

In this lab you will:

- Conduct a parameter search to find the optimal value for K 
- Use a KNN classifier to generate predictions on a real-world dataset 
- Evaluate the performance of a KNN model  


## Getting Started

Start by importing the dataset, stored in the `titanic.csv` file, and previewing it.


```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")

# Preview the dataset
df.head()

```

Great!  Next, you'll perform some preprocessing steps such as removing unnecessary columns and normalizing features.

## Preprocessing the data

Preprocessing is an essential component in any data science pipeline. It's not always the most glamorous task as might be an engaging data visual or impressive neural network, but cleaning and normalizing raw datasets is very essential to produce useful and insightful datasets that form the backbone of all data powered projects. This can include changing column types, as in: 


```python
df['col_name'] = df['col_name'].astype('int')
```
Or extracting subsets of information, such as: 

```python
import re
df['street'] = df['address'].map(lambda x: re.findall('(.*)?\n', x)[0])
```

> **Note:** While outside the scope of this particular lesson, **regular expressions** (mentioned above) are powerful tools for pattern matching! See the [regular expressions official documentation here](https://docs.python.org/3.6/library/re.html). 

Since you've done this before, you should be able to do this quite well yourself without much hand holding by now. In the cells below, complete the following steps:

1. Remove unnecessary columns (`'PassengerId'`, `'Name'`, `'Ticket'`, and `'Cabin'`) 
2. Convert `'Sex'` to a binary encoding, where female is `0` and male is `1` 
3. Detect and deal with any missing values in the dataset:  
    * For `'Age'`, replace missing values with the median age for the dataset  
    * For `'Embarked'`, drop the rows that contain missing values
4. One-hot encode categorical columns such as `'Embarked'` 
5. Store the target column, `'Survived'`, in a separate variable and remove it from the DataFrame  

While we always want to worry about data leakage, which is why we typically perform the split before the preprocessing, for this data set, we'll do some of the preprocessing first. The reason for this is that some of the values of the variables only have a handful of instances, and we want to make sure we don't lose any of them.


```python
# Drop the unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

df.head()
```


```python
# Convert Sex to binary encoding

df = df.copy()
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

df.head()
```


```python
# Find the number of missing values in each column
missing_values = df.isnull().sum()
missing_values
```
```
Survived      0
Pclass        0
Sex           0
Age         177
SibSp         0
Parch         0
Fare          0
Embarked      2
dtype: int64
```


```python
# Impute the missing values in 'Age'
df['Age'] = df['Age'].fillna(df['Age'].mean())
df.isna().sum()
```
```
Survived    0
Pclass      0
Sex         0
Age         0
SibSp       0
Parch       0
Fare        0
Embarked    2
dtype: int64
```


```python
# Drop the rows missing values in the 'Embarked' column
df = None
df.isna().sum()
```


```python
# One-hot encode the categorical columns
one_hot_df = None
one_hot_df.head()
```
```
Pclass   Age  SibSp  Parch     Fare  Sex_1  Embarked_Q  Embarked_S
0       3  22.0      1      0   7.2500      1           0           1
1       1  38.0      1      0  71.2833      0           0           0
2       3  26.0      0      0   7.9250      0           0           1
3       1  35.0      1      0  53.1000      0           0           1
4       3  35.0      0      0   8.0500      1           0           1
```

```python
# Assign the 'Survived' column to labels
labels = df['Survived']

# Drop the 'Survived' column from one_hot_df
df.drop(columns=['Survived'], inplace=True)
```

## Create training and test sets

Now that you've preprocessed the data, it's time to split it into training and test sets. 

In the cell below:

* Import `train_test_split` from the `sklearn.model_selection` module 
* Use `train_test_split()` to split the data into training and test sets, with a `test_size` of `0.25`. Set the `random_state` to 42 


```python
# Import train_test_split 
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```
```
((666, 7), (223, 7), (666,), (223,))
```

## Normalizing the data

The final step in your preprocessing efforts for this lab is to **_normalize_** the data. We normalize **after** splitting our data into training and test sets. This is to avoid information "leaking" from our test set into our training set (read more about data leakage [here](https://machinelearningmastery.com/data-leakage-machine-learning/) ). Remember that normalization (also sometimes called **_Standardization_** or **_Scaling_**) means making sure that all of your data is represented at the same scale. The most common way to do this is to convert all numerical values to z-scores. 

Since KNN is a distance-based classifier, if data is in different scales, then larger scaled features have a larger impact on the distance between points.

To scale your data, use `StandardScaler` found in the `sklearn.preprocessing` module. 

In the cell below:

* Import and instantiate `StandardScaler` 
* Use the scaler's `.fit_transform()` method to create a scaled version of the training dataset  
* Use the scaler's `.transform()` method to create a scaled version of the test dataset  
* The result returned by `.fit_transform()` and `.transform()` methods will be numpy arrays, not a pandas DataFrame. Create a new pandas DataFrame out of this object called `scaled_df`. To set the column names back to their original state, set the `columns` parameter to `one_hot_df.columns` 
* Print the head of `scaled_df` to ensure everything worked correctly 


```python
# Step: Normalize the Data Again

# Instantiate the StandardScaler
scaler = StandardScaler()

# Fit and transform the training set
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test set (without fitting to avoid data leakage)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame with original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Display the first few rows of the normalized training set
X_train_scaled.head()
```

You may have noticed that the scaler also scaled our binary/one-hot encoded columns, too! Although it doesn't look as pretty, this has no negative effect on the model. Each 1 and 0 have been replaced with corresponding decimal values, but each binary column still only contains 2 values, meaning the overall information content of each column has not changed.

## Fit a KNN model

Now that you've preprocessed the data it's time to train a KNN classifier and validate its accuracy. 

In the cells below:

* Import `KNeighborsClassifier` from the `sklearn.neighbors` module 
* Instantiate the classifier. For now, you can just use the default parameters  
* Fit the classifier to the training data/labels
* Use the classifier to generate predictions on the test data. Store these predictions inside the variable `test_preds` 


```python
# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)  # Default k=5


# Instantiate KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)


# Fit the classifier
clf.fit(X_train_scaled, y_train)

# Predict on the test set
test_preds = clf.predict(X_test_scaled)

```

## Evaluate the model

Now, in the cells below, import all the necessary evaluation metrics from `sklearn.metrics` and complete the `print_metrics()` function so that it prints out **_Precision, Recall, Accuracy, and F1-Score_** when given a set of `labels` (the true values) and `preds` (the models predictions). 

Finally, use `print_metrics()` to print the evaluation metrics for the test predictions stored in `test_preds`, and the corresponding labels in `y_test`. 


```python
# Your code here 
# Import the necessary functions
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

```


```python
# Complete the function
def print_metrics(labels, preds):
    print("Precision Score:", precision_score(labels, preds))
    print("Recall Score:", recall_score(labels, preds))
    print("Accuracy Score:", accuracy_score(labels, preds))
    print("F1 Score:", f1_score(labels, preds))

print_metrics(y_test, test_preds)
```
```
Precision Score: 0.7023809523809523
Recall Score: 0.7195121951219512
Accuracy Score: 0.7847533632286996
F1 Score: 0.710843373493976
```

> Interpret each of the metrics above, and explain what they tell you about your model's capabilities. If you had to pick one score to best describe the performance of the model, which would you choose? Explain your answer.

Write your answer below this line: 


________________________________________________________________________________




## Improve model performance

While your overall model results should be better than random chance, they're probably mediocre at best given that you haven't tuned the model yet. For the remainder of this notebook, you'll focus on improving your model's performance. Remember that modeling is an **_iterative process_**, and developing a baseline out of the box model such as the one above is always a good start. 

First, try to find the optimal number of neighbors to use for the classifier. To do this, complete the `find_best_k()` function below to iterate over multiple values of K and find the value of K that returns the best overall performance. 

The function takes in six arguments:
* `X_train`
* `y_train`
* `X_test`
* `y_test`
* `min_k` (default is 1)
* `max_k` (default is 25)
    
> **Pseudocode Hint**:
1. Create two variables, `best_k` and `best_score`
1. Iterate through every **_odd number_** between `min_k` and `max_k + 1`. 
    1. For each iteration:
        1. Create a new `KNN` classifier, and set the `n_neighbors` parameter to the current value for k, as determined by the loop 
        1. Fit this classifier to the training data 
        1. Generate predictions for `X_test` using the fitted classifier 
        1. Calculate the **_F1-score_** for these predictions 
        1. Compare this F1-score to `best_score`. If better, update `best_score` and `best_k` 
1. Once all iterations are complete, print the best value for k and the F1-score it achieved 


```python
# Function to find the best k using F1-score
def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    best_k = None
    best_score = 0
    
    for k in range(min_k, max_k + 1, 2):  # Iterate through odd numbers (to prevent ties)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        
        if f1 > best_score:
            best_score = f1
            best_k = k
    
    print(f"Best Value for k: {best_k}")
    print(f"F1-Score: {best_score:.4f}")
    return best_k

# Find the optimal k
optimal_k = find_best_k(X_train_scaled, y_train, X_test_scaled, y_test)
```
```
Best Value for k: 17
F1-Score: 0.7468354430379746
```

If all went well, you'll notice that model performance has improved by 3 percent by finding an optimal value for k. For further tuning, you can use scikit-learn's built-in `GridSearch()` to perform a similar exhaustive check of hyperparameter combinations and fine tune model performance. For a full list of model parameters, see the [sklearn documentation !](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)




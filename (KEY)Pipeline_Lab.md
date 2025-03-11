# Pipelines in scikit-learn - Lab 

## Introduction 

In this lab, you will work with the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). The goal of this lab is not to teach you a new classifier or even show you how to improve the performance of your existing model, but rather to help you streamline your machine learning workflows using scikit-learn pipelines. Pipelines let you keep your preprocessing and model building steps together, thus simplifying your cognitive load. You will see for yourself why pipelines are great by building the same KNN model twice in different ways. 

## Objectives 

- Construct pipelines in scikit-learn 
- Use pipelines in combination with `GridSearchCV()`

## Import the data

Run the following cell to import all the necessary classes, functions, and packages you need for this lab. 


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')
```

Import the `'winequality-red.csv'` dataset and print the first five rows of the data.  


```python
# Import the data
df = pd.read_csv('winequality-red.csv')


# Print the first five rows
df.head()
```
```
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.4              0.70         0.00             1.9      0.076   
1            7.8              0.88         0.00             2.6      0.098   
2            7.8              0.76         0.04             2.3      0.092   
3           11.2              0.28         0.56             1.9      0.075   
4            7.4              0.70         0.00             1.9      0.076   

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 11.0                  34.0   0.9978  3.51       0.56   
1                 25.0                  67.0   0.9968  3.20       0.68   
2                 15.0                  54.0   0.9970  3.26       0.65   
3                 17.0                  60.0   0.9980  3.16       0.58   
4                 11.0                  34.0   0.9978  3.51       0.56
```

Use the `.describe()` method to print the summary stats of all columns in `df`. Pay close attention to the range (min and max values) of all columns. What do you notice? 

```python
# Print the summary stats of all columns
df.describe()
```
```
       fixed acidity  volatile acidity  citric acid  residual sugar  \
count    1599.000000       1599.000000  1599.000000     1599.000000   
mean        8.319637          0.527821     0.270976        2.538806   
std         1.741096          0.179060     0.194801        1.409928   
min         4.600000          0.120000     0.000000        0.900000   
25%         7.100000          0.390000     0.090000        1.900000   
50%         7.900000          0.520000     0.260000        2.200000   
75%         9.200000          0.640000     0.420000        2.600000   
max        15.900000          1.580000     1.000000       15.500000  
```

As you can see from the data, not all features are on the same scale. Since we will be using k-nearest neighbors, which uses the distance between features to classify points, we need to bring all these features to the same scale. This can be done using standardization. 

However, before standardizing the data, let's split it into training and test sets. 

> Note: You should always split the data before applying any scaling/preprocessing techniques in order to avoid data leakage. If you don't recall why this is necessary, you should refer to the **KNN with scikit-learn - Lab.** 

## Split the data 

- Assign the target (`'quality'` column) to `y` 
- Drop this column and assign all the predictors to `X` 
- Split `X` and `y` into 75/25 training and test sets. Set `random_state` to 42  


```python
# Split the predictor and target variables
y = df['quality']
X = df.drop(columns=['quality'])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

# Split into training and test sets
X_train, X_test, y_train, y_test = None
```

## Standardize your data 

- Instantiate a `StandardScaler()` 
- Transform and fit the training data 
- Transform the test data 


```python
# Instantiate StandardScaler
scaler = StandardScaler()

# Transform the training and test sets
scaled_data_train = scaler.fit_transform(X_train)
scaled_data_test = scaler.transform(X_test)

# Convert into a DataFrame
scaled_df_train = pd.DataFrame(scaled_data_train, columns=X_train.columns)
print(scaled_df_train.head())


# Convert into a DataFrame
scaled_df_train = pd.DataFrame(scaled_data_train, columns=X_train.columns)
scaled_df_train.head()
```

## Train a model 

- Instantiate a `KNeighborsClassifier()` 
- Fit the classifier to the scaled training data 


```python
# Instantiate KNeighborsClassifier
clf = KNeighborsClassifier()

# Fit the classifier
clf.fit(scaled_data_train, y_train)
```
```
KNeighborsClassifier()
```

Use the classifier's `.score()` method to calculate the accuracy on the test set (use the scaled test data) 

```python
# Print the accuracy on test set
accuracy_knn = clf.score(scaled_data_test, y_test)
print(f'KNN Model Accuracy: {accuracy_knn:.4f}')
```
```
KNN Model Accuracy: 0.5775
```

Nicely done. This pattern (preprocessing and fitting models) is very common. Although this process is fairly straightforward once you get the hang of it, **pipelines** make this process simpler, intuitive, and less error-prone. 

Instead of standardizing and fitting the model separately, you can do this in one step using `sklearn`'s `Pipeline()`. A pipeline takes in any number of preprocessing steps, each with `.fit()` and `transform()` methods (like `StandardScaler()` above), and a final step with a `.fit()` method (an estimator like `KNeighborsClassifier()`). The pipeline then sequentially applies the preprocessing steps and finally fits the model. Do this now.   

## Build a pipeline (I) 

Build a pipeline with two steps: 

- First step: `StandardScaler()` 
- Second step (estimator): `KNeighborsClassifier()` 



```python
# Fit the training data to pipeline
scaled_pipeline_1.fit(X_train, y_train)

# Print the accuracy on test set
accuracy = scaled_pipeline_1.score(X_test, y_test)
print(f"Test Set Accuracy: {accuracy:.3f}")
```
```
Test Set Accuracy: 0.578
```

- Transform and fit the model using this pipeline to the training data (you should use `X_train` here) 
- Print the accuracy of the model on the test set (you should use `X_test` here) 


```python
# Fit the training data to pipeline


# Print the accuracy on test set

```

If you did everything right, this answer should match the one from above! 

Of course, you can also perform a grid search to determine which combination of hyperparameters can be used to build the best possible model. The way you define the pipeline still remains the same. What you need to do next is define the grid and then use `GridSearchCV()`. Let's do this now.

## Build a pipeline (II)

Again, build a pipeline with two steps: 

- First step: `StandardScaler()` named 'ss'.  
- Second step (estimator): `RandomForestClassifier()` named 'RF'. Set `random_state=123` when instantiating the random forest classifier 


```python
# Build a pipeline with StandardScaler and RandomForestClassifier
scaled_pipeline_2 = None
```

Use the defined `grid` to perform a grid search. We limited the hyperparameters and possible values to only a few values in order to limit the runtime. 


```python
# Define the grid
grid = [{'RF__max_depth': [4, 5, 6], 
         'RF__min_samples_split': [2, 5, 10], 
         'RF__min_samples_leaf': [1, 3, 5]}]
```

Define a grid search now. Use: 
- the pipeline you defined above (`scaled_pipeline_2`) as the estimator 
- the parameter `grid` 
- `'accuracy'` to evaluate the score 
- 5-fold cross-validation 


```python
# Define a grid search
gridsearch = None
```

After defining the grid values and the grid search criteria, all that is left to do is fit the model to training data and then score the test set. Do this below: 


```python
# Fit the training data


# Print the accuracy on test set

```

## Summary

See how easy it is to define pipelines? Pipelines keep your preprocessing steps and models together, thus making your life easier. You can apply multiple preprocessing steps before fitting a model in a pipeline. You can even include dimensionality reduction techniques such as PCA in your pipelines. In a later section, you will work on this too! 

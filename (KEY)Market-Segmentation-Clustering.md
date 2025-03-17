
# Market Segmentation with Clustering - Lab

## Introduction

In this lab, you'll use your knowledge of clustering to perform market segmentation on a real-world dataset!

## Objectives

In this lab you will: 

- Use clustering to create and interpret market segmentation on real-world data 

## Getting Started

In this lab, you're going to work with the [Wholesale customers dataset](https://archive.ics.uci.edu/ml/datasets/wholesale+customers) from the UCI Machine Learning datasets repository. This dataset contains data on wholesale purchasing information from real businesses. These businesses range from small cafes and hotels to grocery stores and other retailers. 

Here's the data dictionary for this dataset:

|      Column      |                                               Description                                              |
|:----------------:|:------------------------------------------------------------------------------------------------------:|
|       FRESH      |                    Annual spending on fresh products, such as fruits and vegetables                    |
|       MILK       |                               Annual spending on milk and dairy products                               |
|      GROCERY     |                                   Annual spending on grocery products                                  |
|      FROZEN      |                                   Annual spending on frozen products                                   |
| DETERGENTS_PAPER |                  Annual spending on detergents, cleaning supplies, and paper products                  |
|   DELICATESSEN   |                           Annual spending on meats and delicatessen products                           |
|      CHANNEL     | Type of customer.  1=Hotel/Restaurant/Cafe, 2=Retailer. (This is what we'll use clustering to predict) |
|      REGION      |            Region of Portugal that the customer is located in. (This column will be dropped)           |



One benefit of working with this dataset for practice with segmentation is that we actually have the ground-truth labels of what market segment each customer actually belongs to. For this reason, we'll borrow some methodology from supervised learning and store these labels separately, so that we can use them afterward to check how well our clustering segmentation actually performed. 

Let's get started by importing everything we'll need.

In the cell below:

* Import `pandas`, `numpy`, and `matplotlib.pyplot`, and set the standard alias for each. 
* Use `numpy` to set a random seed of `0`.
* Set all matplotlib visualizations to appear inline.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(0)
```

Now, let's load our data and inspect it. You'll find the data stored in `'wholesale_customers_data.csv'`. 

In the cell below, load the data into a DataFrame and then display the first five rows to ensure everything loaded correctly.


```python
raw_df = pd.read_csv('wholesale_customers_data.csv')
raw_df```
```
```
   Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185
```

Now, let's go ahead and store the `'Channel'` column in a separate variable and then drop both the `'Channel'` and `'Region'` columns. Then, display the first five rows of the new DataFrame to ensure everything worked correctly. 


```python
channels = raw['Channel']
df = raw_df.drop(['Channel', 'Region'], axis=1)
```

Now, let's get right down to it and begin our clustering analysis. 

In the cell below:

* Import `KMeans` from `sklearn.cluster`, and then create an instance of it. Set the number of clusters to `2`
* Fit it to the data (`df`) 
* Get the predictions from the clustering algorithm and store them in `cluster_preds` 


```python
from sklearn.cluster import KMeans
```

```python
k_means = KMeans(n_clusters=2, random_state=0)

cluster_preds = k_means.fit_predict(df)
```

Now, use some of the metrics to check the performance. You'll use `calinski_harabasz_score()` and `adjusted_rand_score()`, which can both be found inside [`sklearn.metrics`](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation). 

In the cell below, import these scoring functions. 


```python
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

# Calinski-Harabasz Score measures the variance ratio within clusters. 
# Higher values indicate well-separated and compact clusters.
ch_score = calinski_harabasz_score(df, cluster_preds)

# Adjusted Rand Index (ARI) compares the predicted clusters to the true labels.
# It checks how well the clustering matches the actual segmentation.
# ARI ranges from -1 (worst) to 1 (perfect clustering).
ari_score = adjusted_rand_score(channels, cluster_preds)

# Print the evaluation scores for analysis
print(f'Calinski-Harabasz Score: {ch_score}')  # Higher is better (good cluster separation)
print(f'Adjusted Rand Index Score: {ari_score}')  # Closer to 1 is better (better match with true labels)
```
```
Calinski-Harabasz Score: 171.68461633384186
Adjusted Rand Index Score: -0.03060891241109425
```

Now, start with CH score to get the variance ratio. 

Although you don't have any other numbers to compare this to, this is a pretty low score, suggesting that the clusters aren't great. 

Since you actually have ground-truth labels, in this case you can use `adjusted_rand_score()` to check how well the clustering performed. Adjusted Rand score is meant to compare two clusterings, which the score can interpret our labels as. This will tell us how similar the predicted clusters are to the actual channels. 

Adjusted Rand score is bounded between -1 and 1. A score close to 1 shows that the clusters are almost identical. A score close to 0 means that predictions are essentially random, while a score close to -1 means that the predictions are pathologically bad, since they are worse than random chance. 

In the cell below, call `adjusted_rand_score()` and pass in `channels` and `cluster_preds` to see how well your first iteration of clustering performed. 


```python
```python
# Adjusted Rand Index (ARI) evaluates how well the predicted clusters match the true segment labels.
# It ranges from -1 (worst) to 1 (perfect clustering), with 0 meaning random clustering.
ari_scaled = adjusted_rand_score(channels, cluster_preds)

# Print the ARI score after scaling to compare improvement.
print(f'Adjusted Rand Index after Scaling: {ari_scaled}')  # Higher is better (closer to 1 means better clustering)
```
```
Adjusted Rand Index after Scaling: -0.03060891241109425
```


According to these results, the clusterings were essentially no better than random chance. Let's see if you can improve this. 

### Scaling our dataset

Recall that k-means clustering is heavily affected by scaling. Since the clustering algorithm is distance-based, this makes sense. Let's use `StandardScaler` to scale our dataset and then try our clustering again and see if the results are different. 

In the cells below:

* Import and instantiate [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and use it to transform the dataset  
* Instantiate and fit k-means to this scaled data, and then use it to predict clusters 
* Calculate the adjusted Rand score for these new predictions 


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```


```python
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_preds_scaled = kmeans.fit_predict(df_scaled)

```


```python
ari_scaled = adjusted_rand_score(channels, cluster_preds_scaled)
print(f'Adjusted Rand Index after Scaling: {ari_scaled}')

```
```
Adjusted Rand Index after Scaling: 0.23664708510864038
```

That's a big improvement! Although it's not perfect, we can see that scaling our data had a significant effect on the quality of our clusters. 

## Incorporating PCA

Since clustering algorithms are distance-based, this means that dimensionality has a definite effect on their performance. The greater the dimensionality of the dataset, the greater the total area that we have to worry about our clusters existing in. Let's try using Principal Component Analysis to transform our data and see if this affects the performance of our clustering algorithm. 

Since you've already seen PCA in a previous section, we will let you figure this out by yourself. 

In the cells below:

* Import [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) from the appropriate module in sklearn 
* Create a `PCA` instance and use it to transform our scaled data  
* Investigate the explained variance ratio for each Principal Component. Consider dropping certain components to reduce dimensionality if you feel it is worth the loss of information 
* Create a new `KMeans` object, fit it to our PCA-transformed data, and check the adjusted Rand score of the predictions it makes. 

**_NOTE:_** Your overall goal here is to get the highest possible adjusted Rand score. Don't be afraid to change parameters and rerun things to see how it changes. 

#### 
```python
from sklearn.decomposition import PCA

pca = PCA()
df_pca = pca.fit_transform(df_scaled)

# Check explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

```
```
[0.44082893 0.283764   0.12334413 0.09395504 0.04761272 0.01049519]
```

#### Decide how many components to keep (usually 2-3):
```python
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

kmeans_pca = KMeans(n_clusters=2, random_state=0)
cluster_preds_pca = kmeans_pca.fit_predict(df_pca)

ari_pca = adjusted_rand_score(channels, cluster_preds_pca)
print(f'Adjusted Rand Index after PCA: {ari_pca}')
```
```
Adjusted Rand Index after PCA: 0.23084287036169227
```

**_Question_**:  What was the Highest Adjusted Rand Score you achieved? Interpret this score and determine the overall quality of the clustering. Did PCA affect the performance overall?  How many principal components resulted in the best overall clustering performance? Why do you think this is?

Write your answer below this line:
_______________________________________________________________________________________________________________________________

## Optional (Level up) 

### Hierarchical Agglomerative Clustering

Now that we've tried doing market segmentation with k-means clustering, let's end this lab by trying with HAC!

In the cells below, use [Agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) to make cluster predictions on the datasets we've created and see how HAC's performance compares to k-mean's performance. 

**_NOTE_**: Don't just try HAC on the PCA-transformed dataset -- also compare algorithm performance on the scaled and unscaled datasets, as well! 


```python
from sklearn.cluster import AgglomerativeClustering

agglo = AgglomerativeClustering(n_clusters=2)

# On scaled PCA data:
agglo_preds_pca = agglo.fit_predict(df_pca)
agglo_ari_pca = adjusted_rand_score(channels, agglo_preds_pca)

print(f'Agglomerative Clustering ARI (with PCA): {agglo_ari_pca}')
```
```
Agglomerative Clustering ARI (with PCA): 0.0459127111971714
```

```python

```


```python

```


```python

```


```python

```


```python

```

## Summary

In this lab, you used your knowledge of clustering to perform a market segmentation on a real-world dataset. You started with a cluster analysis with poor performance, and then implemented some changes to iteratively improve the performance of the clustering analysis!

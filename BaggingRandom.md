# Bagging and Random Forest

Decision Tree (Very Sensitive to tree depth and not-robust): 
- Recursevley make splits based on entropy or impuruty
- Split on features best increasing information gain
- Keep splitting until leaf is pure
- Tendency to overfitt at large max depth
- High enouth depth: will fit to training set perfectly
- Decison Valaries:
- Training on Different Realizations: a set of  points 

To avoid: Reduce decision tree depth to limit variance, but it may underfit then.

Criterion is harsh
Culmen Length

Can we figure out a way in a split for a given subregion to factor in different features?
- Recursion: after split on "best" feature, different subsets never talk to each other again
- But maybe other branches/regions: info influencing split/class assignment in given region

Decision Tree's are usually super fast, but also we 

- Bagging (Bootstrap Aggrefation)= way to create realization of dataset with different wights for data. Basically sampling with replacement
- Sampling with replacemnet creates more weight
- Each of weighted samples, each are trained on different traing sets (Allows for to address variance issues, doesnt rely on a specific data point)


Now use ensemble of trained models (Aggregate to make prediction on test data)
- aggrestion function is average of regressor treees
- You take the mean, and then that will be your decision tree prediction

Bagging classifer is a wrapper around other estiamters

Train, Test, Split

baf_class_decsion = BaggingClassifer(estimateor= DecisionTreeClassifer(), n_estimator = 150)
bag_pipe = Pipeline([('scaler', StandardScaler()),
('model'',
bag_class_decision)])
params +

Put it in grid search

Fir the best mmodel. Get 

Is the bastline decsion tree over the same rrange of tree depth>

Bagging is better on precision/recall for all classes --especially positive one

Bagging alone is not enough to capture feature space
bootstrapped samples are highly correlated with each other 


Effectivley factiri in other feature-space when aking decision on class assignments


Bagging Forest and Radom forest both use bagging, but Random Forest only takes m<M for each node for split. Bagging Trees take all M features



Aggregating smoths large fluctutatuins of class assignmetns from individual trees out.
Due to feature subest sampling: can learn more complex boundaries: smoothen these.
Can also get probabilut of class assignment:

- .feature_importance_atribute

Extra Tree: Sometimes our variance problems are extreme.

Threee levels of randomization
- sampling of data
- sampling of features
- random selection

- 

First EDA then Train/Test/Split then make your model, then use metrics

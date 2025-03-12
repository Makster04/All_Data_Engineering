# All_Data_Engineering

Gradient Boosting uses weight to deal with the overfiitng
- Works best on stumps

- bags the data into

Bagging is with replacemtns 
- with replacemnet can be repeats
- without it you cant repeact
  
Adaboosting d (If the weight was one, it would basxially be gradient boosting Adaboost is gradietn boost if Aplhpa =1 ).


Radnom Forest is multiple decision trees


Random Forest uses bagging to work effectivley 


Recomendation

Princoical Component Analysis (PCA) (Coordinate transformation and Projection (to lower dimemsional set)
- Principal Component 1
- Principal Componet 2 (Instead of X and I, I can describe it in new coordinate system, Z1 and Z2)
Id there is one direction, then I can project (Effecitlvley reduced my dimmenstionality to one)


PCA
- you use this oftern with much higher-D spaces
- Vatiance depends oon the scale of the feature so you want to standarfdize before you PCA
- Transfrom coordinates construct new features
- Slope being 0 means theyre not correlated at all

PC0 and PC1 dominate the vraince

After being done with PCA, Orthoognol (They are not correlated with eahcother, meaning they are, 

We take thae ones that dominate the vector call (PC)

We gonna project our data on these three directions

Example:
Projject 5D to 2D (Less dimenstions with more Data)
PCA acts as a trainfoemer in the poepline

fit/transformer scaler



discovered Clusters to help predict 
can become hot encode
Cluster is characterized by its center


Cost-Function of K
inertia

As you converge, it stabilizes


K mean is a distance algorithm
Scale featire befor scaling cluster


Do grid search if yout dont know the number of clusters

.cluster_centers
labels_attirbut

Since the cneters are scaled, need to unscare back to original dat scale
elbow method: Use intertia, 


Silhouette coeeficent: Better wau to mseasure clustering effectivnents

Cohesion: How tightly bound is each point given clusters  to its cluster members? (Should be smallO
Differentiation: How far waya is point of a given cluster pounts in other clusters? (Should be big

Sillutte Index: Combines both 

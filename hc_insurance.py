# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('insurance.csv')
# Taking Age and Price only for the prediction
X = dataset.iloc[:, [0, 6]].values

# Using the Dendogram to find optimal number of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.label('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Eculiance Distance')
#5 Clusture 

# Fitting Hierarchical Clusturing to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity = 'euclidean', linkage='ward')
y_hc=hc.fit_predict(X)

#Visualising the clusters
# If u are using more then 2 dimension don't execute this portion of code as it is valid in 2d only
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Steady')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Intellegent')
"""
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Managsable')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Trustable')
"""
plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()
plt.show()

from numpy import *
from sklearn.neighbors import KNeighborsClassifier

def classifyKNN(trainSet, trainLabels, testSet, n):
    #, weights="distance"
    neigh = KNeighborsClassifier(n_neighbors=n);
    neigh.fit(trainSet, trainLabels);
    
    predictedLabels = neigh.predict(testSet);
    
    return predictedLabels
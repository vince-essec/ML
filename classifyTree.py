from numpy import *
from sklearn import tree

def classifyTree(trainSet, trainLabels, testSet, n):
 
    clf = tree.DecisionTreeClassifier(max_depth = n);
    clf = clf.fit(trainSet, trainLabels);
    predictedLabels = clf.predict(testSet);
    
    return predictedLabels
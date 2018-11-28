from numpy import *

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC  

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import *

from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier





def DecisionTree(trainSet, trainLabels, testSet):

	

    # Train a single decision tree

    classifier = DecisionTreeClassifier(max_depth=7)



    # Train the classifier

    classifier.fit(trainSet, trainLabels)



    # Do classification on the test dataset and return 

    predictedLabels = classifier.predict(testSet)

    

    return predictedLabels





def RandomForest(trainSet, trainLabels, testSet):

    

    classifier = RandomForestClassifier(n_estimators = 450, max_features = 'sqrt', bootstrap = True, min_samples_leaf = 2)

    

    classifier.fit(trainSet, trainLabels)

    

    predictedLabels = classifier.predict(testSet)

    

    return predictedLabels





def LogisticRegression(trainSet, trainLabels, testSet):



    # Train 

    classifier = LogisticRegressionClassifier(C = 1e12, penalty = 'l2', random_state= 33)



    classifier.fit(trainSet,trainLabels)



    predictedLabels = classifier.predict(testSet)



    return predictedLabels





def KNN(trainSet, trainLabels, testSet):

    

    classifier = KNearestNeighborsClassifier()

    

    classifier.fit(trainSet, trainLabels)

    

    predictedLabels = classifier.predict(testSet)

    

    return predictedLabels






def LinearSVM(trainSet, trainLabels, testSet):

    

    classifier = SVC(kernel='linear') 

    

    classifier.fit(trainSet, trainLabels)

    

    ##supportVectors = classifier.support_vectors_ ??

    

    predictedLabels = classifier.predict(testSet)

    

    return predictedLabels




def KMeans(trainSet, trainLabels, testSet):


    classifier  = KMeans(n_clusters=2, random_state=0).fit(X)

    ##kmeans.labels_ ??

    classifier.fit(trainSet, trainLabels)

    classifier.predict(testSet)

    ##kmeans.cluster_centers_ ??

    return predictedLabels




def SpectralClustering(trainSet, trainLabels, testSet):


    classifier = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(X)


    ##clustering.labels_ ??


    classifier.fit(trainSet, trainLabels)


    predictedLabels = classifier.predict(testSet)


    return predictedLabels 





def NeuralNetworks(trainSet, trainLabels, testSet):

    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


    classifier.fit(trainSet, trainLabels)


    predictedLabels = classifier.predict(testSet)


    return predictedLabels 





def GaussianNaïveBayes(trainSet, trainLabels, testSet):


    classifier = GaussianNB()

    

    classifier.fit(trainSet, trainLabels)

    

    predictedLabels = classifier.predict(testSet)

    

    return predictedLabels




###Check Adaboost
def Adaboost(trainSet, trainLabels, testSet):


    classifier = AdaBoostClassifier(n_estimators=100)
    
    scores = cross_val_score(classifier, trainSet, trainLabels, cv=5)
    
    mean_scores = scores.mean() 

    return mean_scores


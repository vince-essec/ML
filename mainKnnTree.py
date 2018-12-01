import os
path = "/Users/vince/Desktop/ESSEC/Master DSBA/T1/Machine Learning (Intro)/Projects/2nd Homework/ML-DSBA-AI-Assignment_2"
os.chdir(path)

from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import classify

path = "/Users/vince/Desktop/ESSEC/Master DSBA/T1/Machine Learning (Intro)/Projects/2nd Homework/ML-DSBA-AI-Assignment_2/Data"
os.chdir(path)

# Load data
csv_file_object = csv.reader(open('titanic_train_new.csv', 'rb')) # Load in the csv file
header = csv_file_object.next() 					  # Skip the fist line as it is a header
data=[] 											  # Create a variable to hold the data

for row in csv_file_object: # Skip through each row in the csv file,
    data.append(row[0:]) 	# adding each row to the data variable
X = array(data) 		    # Then convert from a list to an array.

y = X[:,1].astype(int) # Save labels to y 

X = delete(X,[1,2],1) # Remove survival column from matrix X


# Initialize cross validation
kf = cross_validation.KFold(X.shape[0], n_folds=10)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  

for trainIndex, testIndex in kf:
    trainSet = X[trainIndex]
    testSet = X[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
	
    predictedLabels = classify(trainSet, trainLabels, testSet)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print 'Accuracy: ' + str(float(correct)/(testLabels.size))
    totalCorrect += correct
    totalInstances += testLabels.size
print 'Total Accuracy: ' + str(totalCorrect/float(totalInstances))
	
#%%
# KNN
import os
path = "/Users/vince/Desktop/ESSEC/Master DSBA/T1/Machine Learning (Intro)/Projects/2nd Homework/ML-DSBA-AI-Assignment_2"
os.chdir(path)


from classifyKNN import classifyKNN
import random
import pylab as plt

#Is setting random seed necessary?
random.seed(30)

#From test and learn, this is the selection leading to the best R2
XKnn = delete(X,[0,7,9],1)

accuracyknn = array([])
for i in range(1,200):    
    # Initialize cross validation
    kf = cross_validation.KFold(XKnn.shape[0], n_folds=10)
    
    totalInstances = 0 # Variable that will store the total intances that will be tested 
    totalCorrect = 0 # Variable that will store the correctly predicted intances  
    
    for trainIndex, testIndex in kf:
        trainSet = XKnn[trainIndex]
        testSet = XKnn[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
        
        predictedLabels = classifyKNN(trainSet, trainLabels, testSet, i)
        
        correct = 0	
        for j in range(testSet.shape[0]):
            if predictedLabels[j] == testLabels[j]:
                correct += 1
                
        totalCorrect += correct
        totalInstances += testLabels.size
    accuracyknn = append(accuracyknn, (totalCorrect/float(totalInstances)))
#    print 'Total Accuracy: ' + str(totalCorrect/float(totalInstances))

plt.plot(accuracyknn, label="Accuracy")
plt.legend()
plt.show()

accuracyknn.argmax() # maximum R2 is obtaines at index 23, so for 24 nearest neighbors

from sklearn.metrics import roc_curve, auc

predictedLabels = classifyKNN(trainSet, trainLabels, testSet, 24)

fpr, tpr, _ = roc_curve(testLabels, predictedLabels)
roc_auc=auc(fpr,tpr)
print('roc_auc=',roc_auc)
plt.plot(fpr,tpr,color='darkorange',lw=3,label='ROC_Curve')
plt.plot([0,1],[0,1],color='blue',lw=2,linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')


#%%
# Decision Tree
from classifyTree import classifyTree

kf = cross_validation.KFold(X.shape[0], n_folds=10)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  

random.seed(30)
accuracytree = array([])
for i in range(1,17):
    for trainIndex, testIndex in kf:
        trainSet = XKnn[trainIndex]
        testSet = XKnn[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
        
        predictedLabels = classifyTree(trainSet, trainLabels, testSet, i)
        
        correct = 0	
        for j in range(testSet.shape[0]):
            if predictedLabels[j] == testLabels[j]:
                correct += 1

        totalCorrect += correct
        totalInstances += testLabels.size
    accuracytree = append(accuracytree, (totalCorrect/float(totalInstances)))

plt.plot(accuracytree, label="Accuracy")
plt.legend()
plt.show()

accuracytree.argmax() # maximum R2 is obtaines at index 6, so for 7 as a depth parameter
accuracytree[6]

predictedLabels = classifyTree(trainSet, trainLabels, testSet, 7)

fpr, tpr, _ = roc_curve(testLabels, predictedLabels)
roc_auc=auc(fpr,tpr)
print('roc_auc=',roc_auc)
plt.plot(fpr,tpr,color='darkorange',lw=3,label='ROC_Curve')
plt.plot([0,1],[0,1],color='blue',lw=2,linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')

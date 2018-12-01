#!/usr/bin/env python
# coding: utf-8

# # Titanic Analysis
# ## Data Processing

# In[201]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import style  
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from operator import itemgetter
import re
import warnings
warnings.filterwarnings("ignore")

style.use('seaborn')      
plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.rcParams['axes.unicode_minus'] = False  


# In[202]:


# Load dataset and view the table
titanic = pd.read_csv("/Users/liliang/Desktop/machine learning/Titanic/train.csv")
titanic.head(5).style


# In[203]:


# Examine the data
titanic.describe().style


# In[204]:


#Correlation among variables#
data=titanic[['Survived','Pclass','Age','SibSp','Parch','Fare','Sex']]
data=data.corr()
sns.heatmap(data)
plt.title('train corr')
plt.show()                   
print(data)


# In[205]:


titanic.info() #check missing values#


# In[206]:


titanic[titanic['Embarked'].isnull()] #check other information of passengers whoes embarked lost#


# In[207]:


titanic.Embarked[(titanic.Sex=='female')&(titanic.Pclass==1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title('Survived_Female_Class1')
plt.show()
#the plot shows that s has highest probability#


# In[208]:


#missing value replacement#
#Embarked---replace missing value. the above two passengers have some common features:female,survived,class1#
titanic[titanic['Embarked'].isnull()]
titanic.Embarked[titanic.Embarked.isnull()]=titanic['Embarked'].fillna('S')


# In[209]:


#Cabin#
titanic[titanic['Cabin'].isnull()]


# In[ ]:





# In[210]:


#Age--replace missing value by RFR

# Get the title from the names
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
    
# Verify that we converted everything.
print(pd.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles


# In[211]:


#choose training data to predict age
age_titanic = titanic[['Age','Fare', 'Parch','SibSp','Parch','Title']] #select related variables put into the model
age_titanic_notnull = age_titanic.loc[(titanic['Age'].notnull())] # select nonnull 
age_titanic_isnull = age_titanic.loc[(titanic['Age'].isnull())]
X = age_titanic_notnull.values[:,1:] #set 'Fare', 'Parch','SibSp','Parch','Title' as X#
Y = age_titanic_notnull.values[:,0] # set the first colume 'age' as Y#

# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1) 
#n_estimators : The number of trees in the forest b.
#n_jobs: The number of jobs to run in parallel for both fit and predict.-1 means using all processors. 

RFR.fit(X,Y) #prediction
predictAges = RFR.predict(age_titanic_isnull.values[:,1:])
titanic.loc[titanic['Age'].isnull(), ['Age']]= predictAges
predictAges


# In[212]:


titanic.info() #check whether imputation is successful


# In[213]:


print(titanic["Sex"].unique()) # view

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

print(titanic["Sex"].unique()) # check


# In[214]:


# Replace Embarked label "S" with 0, "C" with 1, "Q" with 2

print(titanic["Embarked"].unique())

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

print(titanic["Embarked"].unique())


# In[215]:


type(titanic)


# In[216]:


# Prepare to test
    # Preprocess the testing datasets (just as what we did previously)
titanic_test = pd.read_csv("/Users/liliang/Desktop/machine learning/Titanic/test.csv")
# titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

# Get all the titles and print how often each one occurs.
titles = titanic_test["Name"].apply(get_title)
print(pd.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Dona":8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
    
# Verify that we converted everything.
print(pd.value_counts(titles))

# Add in the title column.
titanic_test["Title"] = titles


# In[217]:


#Test set--missing value#
#Fare#
titanic_test[titanic_test['Fare'].isnull()]


# In[218]:


fare=titanic_test[(titanic_test['Embarked'] == "S") & (titanic_test['Pclass'] == 3)].Fare.median()
titanic_test['Fare']=titanic_test['Fare'].fillna(fare)


# In[219]:


#Embarked#
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")


# In[220]:




#choose testing data to predict age
age_test = titanic_test[['Age','Fare', 'Parch','SibSp','Parch','Title']]

age_test_notnull = age_test[age_test['Age'].notnull()]
age_test_isnull = age_test[(age_test['Age'].isnull())]

X = age_test_notnull.values[:,1:]
Y = age_test_notnull.values[:,0]


# use RandomForestRegression to test data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predict_test_Ages = RFR.predict(age_test_isnull.values[:,1:])
titanic_test.loc[titanic_test['Age'].isnull(), ['Age']]= predict_test_Ages
predict_test_Ages


# In[221]:


titanic_test.info()


# In[222]:


titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2


# In[223]:


print(titanic_test["Sex"].unique()) # view

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

print(titanic_test["Sex"].unique()) # check


# ## Feature Engineering

# In[224]:


full_dataset = [titanic, titanic_test]


# In[225]:


# Generating age group
for dataset in full_dataset:    
    dataset.loc[dataset["Age"] <= 14, "AgeC"] = 0
    dataset.loc[(dataset["Age"] > 14) & (dataset["Age"] <= 34), "AgeC"] = 1
    dataset.loc[(dataset["Age"] > 34) & (dataset["Age"] <= 53), "AgeC"] = 2
    dataset.loc[dataset["Age"] > 53, "AgeC"] = 3
    
# Generating familysize
for dataset in full_dataset:  
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"]

# Generating traveling alone
for dataset in full_dataset:   
    dataset["Alone"] = 0
    dataset.loc[dataset["FamilySize"] == 0, "Alone"] = 1

# Get the title from the names
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 4, "Major": 4, "Col": 4, "Mlle": 4, "Mme": 4, "Don": 4,"Dona": 4, "Lady": 4, "Countess": 4, "Jonkheer": 4, "Sir": 4, "Capt": 4, "Ms": 4}
    
# Generating Title
for dataset in full_dataset: 
    dataset["Title"] = dataset["Name"].apply(get_title)
    for k,v in title_mapping.items():
        dataset.loc[dataset["Title"] == k, "Title"] = v
          
# Generating Name Length
for dataset in full_dataset: 
    dataset["NameLength"] = dataset["Name"].apply(lambda x: len(x))


# In[226]:


data=['']


# In[227]:


# Generating Price Range
for dataset in full_dataset:    
    dataset.loc[dataset["Fare"] <= 7.9, "FareC"] = 0
    dataset.loc[(dataset["Fare"] > 7.9) & (dataset["Fare"] <= 14.5), "FareC"] = 1
    dataset.loc[(dataset["Fare"] > 14.5) & (dataset["Fare"] <= 31), "FareC"] = 2
    dataset.loc[dataset["Fare"] > 31, "FareC"] = 3


# In[228]:


dataset["FareC"] = dataset["FareC"].astype('int')


# In[229]:


# Generating Has Cabin
for dataset in full_dataset:
    dataset["NoCabin"] = pd.isna(dataset["Cabin"])

# Generating Position
# Map each Position to an integer.
position_mapping = {"0": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}


# In[230]:


for dataset in full_dataset:
    dataset["Position"] = dataset["Cabin"].str[0]
    dataset["Position"].fillna("0", inplace=True) 
    dataset["Position"] = [position_mapping[x] for x in dataset["Position"].astype(str) if x in position_mapping]


# In[231]:


# Generating Pair
def NumberFromStrings(string):
    numberfromstring = re.findall('\d+',string)
    if numberfromstring == []:
        return int(0)
    else:
        return int(numberfromstring[0])

# This method considers cabin without numbers, f.ex. "F", as being even number rooms.
for dataset in full_dataset: 
    dataset["Pair"] = dataset["Cabin"].astype(str)
    dataset["Pair"] = dataset["Pair"].apply(NumberFromStrings)
    dataset["Pair"] = dataset["Pair"] % 2
    dataset.loc[pd.isna(dataset["Cabin"]), "Pair"] = 2


# In[232]:


titanic.Position.unique()


# In[233]:


df = pd.DataFrame({'col2': {0: 'a', 1: 2, 2: np.nan}, 'col1': {0: 'w', 1: 'A', 2: 'B'}})
di = {"B": 2, "A": 1}
print(df)
print(df.replace({"col1": di}))


# In[234]:


titanic.drop(['PassengerId','Name', 'Ticket','Cabin'],axis=1,inplace=True)


# In[237]:


#scale the data#
titanic['Fare'] = StandardScaler().fit_transform(titanic.filter(['Fare']))
titanic['Age'] = StandardScaler().fit_transform(titanic.filter(['Age']))
titanic['NameLength'] = StandardScaler().fit_transform(titanic.filter(['NameLength']))
titanic_test['Fare'] = StandardScaler().fit_transform(titanic_test.filter(['Fare']))
titanic_test['Age'] = StandardScaler().fit_transform(titanic_test.filter(['Age']))
titanic_test['NameLength'] = StandardScaler().fit_transform(titanic_test.filter(['NameLength']))


# In[238]:


titanic


# In[239]:


titanic_test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
titanic_test


# In[240]:


titanic.to_csv('/Users/liliang/Desktop/machine learning/Titanic/titanic_train_scaled.csv')
titanic_test.to_csv('/Users/liliang/Desktop/machine learning/Titanic/titanic_test_scaled.csv')


# In[ ]:





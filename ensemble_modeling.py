#!/usr/bin/env python
# coding: utf-8

# In[249]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
get_ipython().run_line_magic('matplotlib', 'inline')

#from collection import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')


# In[250]:


get_ipython().system('pip install collections')


# In[314]:


#Load DATA
train = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\kaggle\titanic_ensemble_modeling\train.csv')
test = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\kaggle\titanic_ensemble_modeling\test.csv')
gend = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\kaggle\titanic_ensemble_modeling\gender_submission.csv')

IDtest = test['PassengerId']
train.count()


# In[315]:


len(gend)


# In[288]:


#search for outliers (Tukey method)
def search_outliers(df, n, features):
    index_outliers = []
    for col in features:
        quantile_1 = np.percentile(df[col], 25) #25%
        quantile_3 = np.percentile(df[col], 75) #75%
        IQR = quantile_3 - quantile_1 #confidence range
        
        outlier_list_index = df[(df[col]<quantile_1-(1.5*IQR)) | (df[col]>quantile_3+(1.5*IQR))].index
        index_outliers.extend(outlier_list_index)
    
    index_outliers = collections.Counter(index_outliers)    
    index_outliers = list(k for k,v in index_outliers.items() if v>n)
    return index_outliers
        
outliers_drop = search_outliers(train, 1, ['Age', 'SibSp', 'Parch', 'Fare'])        
#train.iloc[outliers_drop].count()


# In[289]:


train = train.drop(outliers_drop, axis = 0).reset_index(drop=True)
train.count()


# In[290]:


train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
dataset.head(100)


# In[291]:


#check null and missing values
dataset = dataset.fillna(np.nan)
dataset.isnull().sum()


# In[292]:


#train.info()
#train.isnull().sum()
train.describe()


# In[293]:


g = sns.heatmap(train[['Survived','SibSp','Parch','Age','Fare']].corr(),cmap='coolwarm',annot=True)


# In[294]:


g = sns.factorplot(x='SibSp', y='Survived', data=train, king='bar', 
                   size=6, palette = 'muted')
g.despine(left=True)
g = g.set_ylabels('survival probability')


# In[295]:


g = sns.factorplot(x='Parch', y='Survived', data=train, king='bar', 
                   size=6, palette = 'muted')
g.despine(left=True)
g = g.set_ylabels('survival probability')


# In[296]:


g=sns.FacetGrid(train, col='Survived')
g=g.map(sns.distplot, 'Age')


# In[297]:


g=sns.kdeplot(train['Age'][(train['Survived']==0)&(train['Age'].notnull())], color='Red',shade=True)
g = sns.kdeplot(train['Age'][(train['Survived']==1)&(train['Age'].notnull())], ax = g, color='Blue', shade = True)
g.set_ylabel('Frequency')
g.set_xlabel('Age')
g = g.legend(['Not Survived', 'Survived'])


# In[298]:


#Categorical values
#Sex
g = sns.barplot(x='Sex', y='Survived', data=train)
g = g.set_ylabel('Survival Probability')


# In[299]:


train[['Sex','Survived']].groupby('Sex').mean()


# In[300]:


#Pclass
g = sns.factorplot(x='Pclass', y='Survived', 
                   data=train, kind='bar', size=6, palette='muted')
g.despine(left=True)
g = g.set_ylabels('survived probability')


# In[301]:


g = sns.factorplot(x='Pclass', y='Survived', hue='Sex', 
                   data=train, kind='bar', size=6, palette='muted')
g.despine(left=True)
g = g.set_ylabels('survived probability')


# In[302]:


#Embarked
dataset['Embarked'] = dataset['Embarked'].fillna('S')
g = sns.factorplot(x='Embarked', y='Survived', 
                   data=train, kind='bar', size=6, palette='muted')
g.despine(left=True)
g = g.set_ylabels('survived probability')


# In[303]:


#Filling missing
age_index_NAN = list(dataset['Age'][dataset['Age'].isnull()].index)
age_med = dataset['Age'].median() 

for i in age_index_NAN:
    age_med = dataset['Age'].median()
    age_pred = dataset['Age'][((dataset['SibSp']==dataset.iloc[i]['SibSp'])
                              &(dataset['Parch']==dataset.iloc[i]['Parch'])
                              &(dataset['Pclass']==dataset.iloc[i]['Pclass']))].median()
    
    #print(age_med)
    #print(age_pred)
    #print('\n')
    
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med
        
g = sns.factorplot(x='Survived', y='Age', data=train, kind='box')


# In[304]:


dataset["Name"].head()


# In[305]:


#Feature engineering
#1) name
dataset_title = [i.split(',')[1].split('.')[0].strip() for i in dataset["Name"]]
dataset['Title'] = pd.Series(dataset_title)
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countes', 'Capt', 'Col',
                            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].map({'Master':0, 'Miss':1, 'Ms':1, 'Mme':1, 'Mlle':1, 'Mrs':1,
                                         'Mr':2, 'Rare':3})
dataset['Title'] = dataset['Title'].astype(int)
dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1})
dataset.drop(labels = ['Name'], axis = 1, inplace = True)
#dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[306]:


#2) Fsize (family size)
dataset['Fsize'] = dataset['SibSp']+dataset['Parch']+1
#create new features
dataset['F_Single'] = dataset['Fsize'].map(lambda s: 1 if s==1 else 0)
dataset['F_Small'] = dataset['Fsize'].map(lambda s: 1 if s==2 else 0)
dataset['F_Med'] = dataset['Fsize'].map(lambda s: 1 if  2<s<5 else 0)
dataset['F_Large'] = dataset['Fsize'].map(lambda s: 1 if s>4 else 0)


# In[307]:


dataset = pd.get_dummies(dataset, columns=['Title'])
dataset = pd.get_dummies(dataset, columns=['Embarked'])


# In[308]:


dataset.drop(labels = ['Ticket', 'Cabin'], axis=1, inplace=True)


# In[309]:


dataset.drop(labels=['PassengerId'], axis=1, inplace=True)


# In[324]:


#dataset.drop(labels=['Title'], axis=1, inplace=True)


# In[317]:


#Modeling
train = dataset[:len(train)]
X_test = dataset[len(train):]
Y_test = X_test['Survived']
X_test.drop(labels=['Survived'], axis=1, inplace=True)
#train['Survived'] = train['Survived'].astype(int)
Y_train = train['Survived']
X_train = train.drop(labels=['Survived'], axis=1)


# In[318]:


X_test.head()


# In[185]:


kfold = StratifiedKFold(n_splits=10)
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), 
                                      random_state=random_state, learning_rate=0.2))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())


# In[204]:


cv_results =[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, Y_train, scoring='accuracy',
                                      cv=kfold, n_jobs=4))

cv_means = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())


# In[205]:


cv_res = pd.DataFrame({'CrossValMeans':cv_means, 'AlgoClassif':['SVC','DecisionTree','AdaBoost',
'RandomForest','ExtraTrees','GradientBoosting','MultipleLayerPerceptron','KNeighboors',
'LogisticRegression','LinearDiscriminantAnalysis']})
g = sns.barplot('CrossValMeans', 'AlgoClassif', data=cv_res, orient='h')


# In[220]:


#hyperparameter
#1
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              'learning_rate':[1.5, 0.5, 0.25, 0.1, 0.01, 0.001]}
                 
gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold,
                        scoring='accuracy', n_jobs=4, verbose=1)

gsadaDTC.fit(X_train, Y_train)
ada_best = gsadaDTC.best_estimator_


# In[222]:


print(ada_best)
print(gsadaDTC.best_score_)


# In[228]:


#2
RandFor = RandomForestClassifier()

RF_param_frid = {"max_depth": [1, 2], "max_features": [1, 3, 10], 
                 "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10], 
                 "bootstrap": [False], "n_estimators" :[100,300], "criterion": ["gini", "entropy"]}

gsRandFor = GridSearchCV(RandFor, param_grid = RF_param_frid, cv=kfold, scoring='accuracy',
                        n_jobs=4, verbose=1)

gsRandFor.fit(X_train, Y_train)
RF_best = gsRandFor.best_estimator_


# In[229]:


print(RF_best)
print(gsRandFor.best_score_)


# In[232]:


GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"], 'n_estimators' : [100,200,300], 'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8], 'min_samples_leaf': [100,150], 'max_features': [0.3, 0.1]}

gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring='accuracy',
                     n_jobs=4, verbose=1)

gsGBC.fit(X_train, Y_train)
GBC_best = gsGBC.best_estimator_


# In[233]:


print(GBC_best)
print(gsGBC.best_score_)


# In[237]:


LogReg = LogisticRegression()

LG_param_grid = {'random_state':[3, 5, 7], 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter':[1000]}

gsLG = GridSearchCV(LogReg, param_grid=LG_param_grid, cv=kfold, scoring='accuracy',
                   n_jobs=4, verbose=1)

gsLG.fit(X_train, Y_train)
LG_best = gsLG.best_estimator_


# In[238]:


print(LG_best)
print(gsLG.best_score_)


# In[241]:


MLPC = MLPClassifier()

mlp_param_grid = {'solver': ['sgd', 'adam', 'lbfgs'], 'activation': ['relu', 'tanh'],
                 'hidden_layer_sizes':[100, 150, 200], 'max_iter':[200, 400]}

gsMLP = GridSearchCV(MLPC, param_grid=mlp_param_grid, cv=kfold, scoring='accuracy',
                     n_jobs=-1, verbose=1)

gsMLP.fit(X_train, Y_train)
MLP_best = gsMLP.best_estimator_


# In[243]:


print(MLP_best)
print(gsMLP.best_score_)


# In[242]:


#ensemble modeling
EnseModel=VotingClassifier(estimators=[('adac', ada_best),('gbc', GBC_best),('mlp', MLP_best),
                                       ('lg', LG_best,),('rfc',RF_best)], voting='soft', n_jobs=-1)

EnseModel = EnseModel.fit(X_train, Y_train)


# In[362]:


#predict = EnseModel.predict(X_test)
#dataset = dataset.fillna(np.nan)
X_test.isnull().sum()


# In[363]:


nan_rows = X_test[X_test['Fare'].isnull()].index
nan_rows


# In[377]:


X_test = X_test.drop(nan_rows, axis=0)


# In[380]:


#X_test.head(75)


# In[379]:


predict = EnseModel.predict(X_test)


# In[382]:


#predict


# In[ ]:





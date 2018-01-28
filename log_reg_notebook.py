
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import os
import sys


# ### Read csv file using pandas

# In[2]:

train = pd.read_csv("train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

labels = train['target']
train.drop(['target', 'id'], axis=1, inplace=True)

print(train.head())


# In[6]:

sss = StratifiedShuffleSplit(labels, test_size=0.05, random_state=1234)
for train_index, test_index in sss:
    break

train_x, train_y = train.values[train_index], labels.values[train_index]
test_x, test_y = train.values[test_index], labels.values[test_index]


# In[4]:

test = pd.read_csv("test.csv")
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

test.drop(['id'], axis=1, inplace=True)

print(test.head())


# ### Tune hyperparameters of logistic regression classifier using gridSearch with 2-fold cross validation

# In[38]:

logreg = LogisticRegression()
param_grid = [{'C': np.logspace(-4, 4, 6),
               'penalty': ['l2', 'l1'],
               'class_weight': [None, 'auto'],
               'solver': ['newton-cg', 'lbfgs', 'liblinear'],
               'multi_class': ['ovr']},
              {'C': np.logspace(-4, 4, 6),
               'penalty': ['l2', 'l1'],
               'class_weight': [None, 'auto'],
               'solver': ['lbfgs'],
               'multi_class': ['multinomial']}]

gs_cv = GridSearchCV(logreg, param_grid, cv = 3, scoring = 'log_loss')
gs_cv.fit(train, labels)


# In[39]:

gs_cv.best_params_                     # hyperparameters which yield the best cv result


# In[42]:

gs_cv.score(train, labels)


# In[41]:

grid_scores = gs_cv.grid_scores_      # scores of all the combinations of hyperparameters tested
grid_scores


# ### Rearrange data to properly save it into a .csv file

# In[43]:

mean, std = [], []
penalty, multi_class, C, solver, class_weight = [], [], [], [], []
for score in grid_scores:
    mean.append(score.mean_validation_score)
    std.append(np.std(score.cv_validation_scores))
    penalty.append(score.parameters['penalty'])
    multi_class.append(score.parameters['multi_class'])
    C.append(score.parameters['C'])
    solver.append(score.parameters['solver'])
    class_weight.append(score.parameters['class_weight'])
    


# In[44]:

df = pd.DataFrame({'mean': mean,
                   'std': std,
                   'penalty': penalty,
                   'multi_class': multi_class,
                   'C': C,
                   'solver': solver,
                   'class_weight': class_weight})
df


# In[45]:

df.to_csv("grid_search_1.csv")


# ### Narrow down search of optimal hyperparameters and increase cross-validation folds

# In[7]:

logreg = LogisticRegression()
param_grid = {'C': np.logspace(-2, 1, 15),
               'penalty': ['l2'],
               'solver': ['newton-cg', 'lbfgs', 'liblinear'],
               'multi_class': ['ovr']}

gs_cv = GridSearchCV(logreg, param_grid, cv = 5, scoring = 'log_loss')
gs_cv.fit(train, labels)


# In[8]:

gs_cv.best_params_                     # hyperparameters which yield the best cv result


# In[10]:

gs_cv.score(train, labels)


# In[11]:

grid_scores = gs_cv.grid_scores_      # scores of all the combinations of hyperparameters tested
grid_scores


# In[35]:

mean, std = [], []
penalty, multi_class, C, solver = [], [], [], []
for score in grid_scores:
    mean.append(score.mean_validation_score)
    std.append(np.std(score.cv_validation_scores))
    penalty.append(score.parameters['penalty'])
    multi_class.append(score.parameters['multi_class'])
    C.append(score.parameters['C'])
    solver.append(score.parameters['solver'])    


# In[36]:

df = pd.DataFrame({'mean': mean,
                   'std': std,
                   'penalty': penalty,
                   'multi_class': multi_class,
                   'C': C,
                   'solver': solver})
df


# In[37]:

df.to_csv("grid_search_2.csv")


# ### using LogisticRegressionCV does not lead to any significance improvement in performance

# In[3]:

cs = np.logspace(-1,1,10)
log_reg1 = LogisticRegressionCV(Cs=cs, cv=5, scoring='log_loss', solver='lbfgs')
log_reg2 = LogisticRegressionCV(Cs=cs, cv=5, scoring='log_loss', solver='newton-cg')
log_reg3 = LogisticRegressionCV(Cs=cs, cv=5, scoring='log_loss', solver='liblinear')


# In[4]:

log_reg1.fit(train, labels)
log_reg2.fit(train, labels)
log_reg3.fit(train, labels)


# In[32]:

cf1 = confusion_matrix(labels, log_reg1.predict(train))
cf2 = confusion_matrix(labels, log_reg2.predict(train))
cf3 = confusion_matrix(labels, log_reg3.predict(train))


# In[35]:

cf3          # predictions wrong!


# ## Use of bagging

# In[9]:

from sklearn.ensemble import BaggingClassifier


# In[10]:

log_reg = LogisticRegression(C=1.3894954943731375, multi_class='ovr', penalty='l2', solver='newton-cg')


# In[11]:

bag_c = BaggingClassifier(base_estimator=log_reg, n_estimators=3, max_samples=0.6, random_state=9876)
bag_c.fit(train, labels)


# In[13]:

score_bag = log_loss(labels, bag_c.predict_proba(train))
cf4 = confusion_matrix(labels, bag_c.predict(train))
print score_bag
print cf4


# In[16]:

n_est = [10, 20, 30, 40, 50, 60, 70, 90, 100]       # number of estimators
max_s = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# In[ ]:

bag_c = BaggingClassifier(base_estimator=log_reg, n_estimators=n, max_samples=m)
bag_c.fit(train_x, train_y)
train_score = log_loss(train_y, bag_c.predict_proba(train_x))
test_score = log_loss(test_y, bag_c.predict_proba(test_x))
res.append([n, m, train_score, test_score])
print train_score, test_score


# In[74]:

param_grid = {'n_estimators': np.arange(1,3,1),
               'max_samples': np.arange(0.7, 0.8, 0.1)}
gs_bag = GridSearchCV(bag_c, param_grid, cv = 2, scoring = 'log_loss')
gs_bag.fit(train, labels)


# In[77]:

gs_bag.grid_scores_


# In[8]:

import sklearn.cross_validation
# print(dir(sklearn.cross_validation))
# help(sklearn.cross_validation)
get_ipython().magic(u'pinfo2 sklearn.cross_validation')


# In[ ]:




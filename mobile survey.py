# -*- coding: utf-8 -*-
"""E:\ssn\z
Created on Thu Nov  7 13:37:55 2019

@author: Tharun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


##predict the current phone 
##

data = pd.read_csv('Survey on Mobile Phone Preferences - Copy.csv')

data['current brand']=data['current brand'].map({'Apple':0,'Asus':1,'honor':2,'HTC':3,'huawei':4,
    'Lenovo':5,'LG':6,'xioami':7,'Motorola':8,'Nokia':9,'Oneplus':10,'Oppo':11,'samsung':12,
    'Realme':13,'Vivo':14})

data = pd.get_dummies(data,columns=['customer Gender','feature','mode of buying',
                      'inducing factors','exchange policy'],drop_first=True)

data['current brand'].value_counts()

x = data.drop('current brand',axis=1)
y = data['current brand']

'''from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)'''


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05,random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

'''from sklearn.model_selection import cross_val_score
acc = cross_val_score(gnb,x_train,y_train,cv=10)'''

y_train.nunique()

acc.mean()
acc.std()

from sklearn import metrics 
metrics.accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test,y_pred)

'''test_size=0.45--->0.18181818181818182
test_size=0.40--->0.15789473684210525
test_size=0.35--->0.058823529411764705
test_size=0.35--->0.06666666666666667
test_size=0.35--->0.08333333333333333
test_size=0.20--->0.1
test_size=0.15--->0.125
test_size=0.10--->0.4
test_size=0.05--->0.666'''
# =============================================================================
# =============================================================================
# # random forest
# =============================================================================
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Survey on Mobile Phone Preferences - Copy.csv')

data['current brand']=data['current brand'].map({'Apple':0,'Asus':1,'honor':2,'HTC':3,'huawei':4,
    'Lenovo':5,'LG':6,'xioami':7,'Motorola':8,'Nokia':9,'Oneplus':10,'Oppo':11,'samsung':12,
    'Realme':13,'Vivo':14})

data = pd.get_dummies(data,columns=['customer Gender','feature','mode of buying',
                               'inducing factors','exchange policy'],drop_first=True)

data['current brand'].value_counts()

x = data.drop('current brand',axis=1)
y = data['current brand']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20,criterion = 'entropy',random_state=0)

classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

accuracy = (cm[0,0]+cm[1,1]+cm[2,2])/(cm[0,0]+cm[1,1]+cm[2,2]+cm[0,1]+cm[0,2]+cm[1,0]
            +cm[1,2]+cm[2,0]+cm[2,1])

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred)*100)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(classifier,x_train,y_train,cv=10)

print(acc.mean())
print(acc.std())

'''


import pydotplus

from sklearn.tree import export_graphviz

dot=export_graphviz(classifier,out_file=None,filled=True,rounded=True)

graph=pydotplus.graph_from_dot_data(dot)
graph.write_png('sample.png')# =============================================================================

# to fing the parameters
# =============================================================================
from sklearn.model_selection import GridSearchCV

param_grid={'bootstrap':[True],'n_estimators':[10,20,50,100]}

classifier_grid= RandomForestClassifier(random_state=0)

gr=GridSearchCV(classifier_grid,param_grid,cv=10,n_jobs=-1)
gr.fit(x_train,y_train)
print(gr.best_params_)

print(gr.best_estimator_)

# =============================================================================
# xgboost-->boosting
# =============================================================================
from xgboost.sklearn import XGBClassifier
classifier1=XGBClassifier()
classifier1.fit(x_train,y_train)

y_pred = classifier1.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

accuracy_xgb = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1])

print(accuracy)
print(accuracy_xgb)
# =============================================================================
# apriori algorithm
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

data = pd.read_excel('Association.xlsx')
data.head()

#In this dataset there is no header row. But by default, pd.read_csv function treats first row as header. 
#To get rid of this problem, add header=None option

records = []
for i in range(0,47):
    records.append([str(data.values[i,j]) for j in range(0, 2)])
    
association_rules=apriori(records, min_support=0.05, min_confidence=0.5, min_lift=1, min_length=2)
#Min_Length =No of items in Rule

association_results=list(association_rules)
print(len(association_results))
print(association_results)

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
    
'''Rule: Apple Old -> Apple
Support: 0.0851063829787234
Confidence: 0.5714285714285714
Lift: 4.476190476190476
=====================================
Rule: Xiaomi -> Xiaomi Old
Support: 0.14893617021276595
Confidence: 0.5833333333333334
Lift: 2.4924242424242427'''

 

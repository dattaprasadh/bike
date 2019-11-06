#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import os as os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost
from xgboost import XGBRegressor


# In[67]:


filePath='/cxldata/datasets/project/bikes.csv'
bikesData=pd.read_csv(filePath)
print(len(bikesData.columns))


# In[7]:


bikesData['yr'].value_counts()


# In[73]:


columnsToDrop = ['instant','casual','registered','atemp','dteday']
bikesData =bikesData.drop(columnsToDrop,1)


# In[90]:


range(bikesData.shape[0])


# In[84]:


columnsToScale=['temp','hum','windspeed']
scaler=StandardScaler()
bikesData[columnsToScale]=scaler.fit_transform(bikesData[columnsToScale])
bikesData['dayCount'] = pd.Series(range(bikesData.shape[0]))/24
bikesData[columnsToScale].describe()


# In[10]:


len(bikesData)


# In[91]:


from sklearn.model_selection import train_test_split

train_set,test_set=train_test_split(bikesData,test_size=0.3,random_state=42)


# In[12]:



from sklearn.model_selection import train_test_split
train_set.sort_values('dayCount', axis= 0, inplace=True)
test_set.sort_values('dayCount', axis= 0, inplace=True)



# In[13]:


len(train_set)


# In[14]:


len(test_set)


# In[15]:


from sklearn.model_selection import train_test_split
train_set.sort_values('dayCount', axis= 0, inplace=True)
test_set.sort_values('dayCount', axis= 0, inplace=True)
train_set


# In[16]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    


# In[17]:


trainingCols=train_set.drop(['cnt'],axis=1)
trainingLabels=train_set['cnt']


# In[18]:


from  sklearn.tree import DecisionTreeRegressor
dec_reg=DecisionTreeRegressor(random_state=42)
from sklearn.model_selection import cross_val_score
cross_val_score(dec_reg,trainingCols, trainingLabels, cv=10,scoring='neg_mean_absolute_error')
dt_mse_scores =np.sqrt(-cross_val_score(dec_reg,trainingCols, trainingLabels, cv=10,scoring='neg_mean_absolute_error'))
dt_mse_scores
display_scores(dt_mse_scores)


# In[19]:


from  sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
from sklearn.model_selection import cross_val_score
cross_val_score(lin_reg,trainingCols, trainingLabels, cv=10,scoring='neg_mean_absolute_error')
lr_mse_scores =np.sqrt(-cross_val_score(lin_reg,trainingCols, trainingLabels, cv=10,scoring='neg_mean_absolute_error'))
lr_mse_scores
display_scores(lr_mse_scores)


# In[20]:


from sklearn.ensemble import RandomForestRegressor


# In[21]:


forest_reg=RandomForestRegressor(random_state=42)
cross_val_score(forest_reg,trainingCols, trainingLabels, cv=10,scoring='neg_mean_absolute_error')
rf_mse_scores =np.sqrt(-cross_val_score(forest_reg,trainingCols, trainingLabels, cv=10,scoring='neg_mean_absolute_error'))
rf_mse_scores
display_scores(rf_mse_scores)


# In[22]:


from xgboost import XGBRegressor
xgb_reg=XGBRegressor(random_state=42,n_estimators=150)
cross_val_score(xgb_reg,trainingCols, trainingLabels, cv=10,scoring='neg_mean_absolute_error')
xg_mae_scores =np.sqrt(-cross_val_score(xgb_reg,trainingCols, trainingLabels, cv=10,scoring='neg_mean_absolute_error'))
display_scores(xg_mae_scores)


# In[23]:


from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [40, 100, 120, 150], 
     'max_features': [8, 10, 12], 'max_depth': [15, 28]
    }]
grid_search=GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_absolute_error')


# In[24]:


j=grid_search.fit(trainingCols,trainingLabels)


# In[25]:


j.best_params_


# In[26]:


final_model=RandomForestRegressor(random_state=42,max_depth= 28,max_features= 12,n_estimators=150)
final_model.fit(trainingCols,trainingLabels)


# In[52]:


test_set.sort_values('dayCount', axis= 0, inplace=True)
   
X_test=test_set.drop(['cnt'],axis=1)
test_x_cols=test_set.drop(['cnt'], axis=1).columns.values
y_test=test_set['cnt']
test_x_cols


# In[28]:


test_set.loc[:,'predictedCounts_test']=final_model.predict(X_test)


# In[31]:


test_set
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,test_set.loc[:,'predictedCounts_test'])
mse
final_mse=np.sqrt(mse)
print (final_mse)


# In[58]:


times = [9,80]
for time in times:
      fig = plt.figure(figsize=(8, 6))
      fig.clf()
      ax = fig.gca()
      test_set_freg_time = test_set[test_set.hr == time]
      test_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'cnt', ax = ax)
      test_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'predictedCounts_test', ax =ax)
      plt.show()


# In[ ]:





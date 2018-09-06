# -*- coding: utf-8 -*-
"""
Created on Sat Aug  25 13:45:40 2018

@author: Pratik
"""

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# We will combine the train and test data to perform feature engineering

train['source'] = 'train'
test['source'] = 'test'

data = pd.concat([train,test], ignore_index=True)

print(train.shape, test.shape, data.shape)

# As the problem is already defined -- we know that we need to predict sales by the store

data.info()
data.describe()

# Some observations
# 1. item_visibility has min value of 0 which is less likely
# 2. Outlet_Establishment_Year will be more useful in a way by which we could know how old it is

#Lets check how many unique items each column has
data.apply(lambda x: len(x.unique()))

# Let us have a look at the object datatype columns

for i in train.columns:
    if train[i].dtype == 'object':
        print(train[i].value_counts())
        print('------------------------------------------------------')
        
#The output gives us following observations:
        
#Item_Fat_Content: Some of ‘Low Fat’ values mis-coded as ‘low fat’ and ‘LF’. Also, some of ‘Regular’ are mentioned as ‘regular’.
#Item_Type: Not all categories have substantial numbers. It looks like combining them can give better results.
#Outlet_Type: Supermarket Type2 and Type3 can be combined. But we should check if that’s a good idea before doing it.

#missing value percentage
#Item_Weight and Outlet_Size has some missing values
print((data[data['Item_Weight'].isnull()].shape[0] / data.shape[0])*100)
print((data[data['Outlet_Size'].isnull()].shape[0] / data.shape[0])*100)

#we impute missing values
data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())
#data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].mode())
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)


#lets change item_visibility from 0 to mean to make sense
data['Item_Visibility']=data['Item_Visibility'].replace(0,data['Item_Visibility'].mean())

#we will calculate meanRatio for each object's visibility
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']],axis=1)
print (data['Item_Visibility_MeanRatio'].describe())

#Now we will categorize the item types as it might be useful for our analysis
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
print(data['Item_Type_Combined'].value_counts())

#Instead of year of establishment, we will consider outlet years in the market
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

#There are typos in Item_Fat_Content variable so we change categories of low fat
print(data['Item_Fat_Content'].value_counts())
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
print(data['Item_Fat_Content'].value_counts())

#mark non consumables as a separate category
data.loc[data['Item_Type_Combined']=="NC",'Item_Fat_Content'] = "Non-Edible"
print(data['Item_Fat_Content'].value_counts())

#Encoding of categorical variables
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = preprocessing.LabelEncoder()
for i in var_mod:
    le = preprocessing.LabelEncoder()
    data[i] = le.fit_transform(data[i])
    
#One Hot Encoding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])

#drop manually converted columns
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#avoid getting in to dummy variable trap
#data = data.drop(['Item_Fat_Content_0','Outlet_Location_Type_0','Outlet_Size_0','Item_Type_Combined_0','Outlet_Type_0','Outlet_0'],axis=1)

#divide in to test and train dataset
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

#drop unnecessary columns
train = train.drop(['source'],axis=1)
test = test.drop(['source','Item_Outlet_Sales'],axis=1)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)

#Some visualizations 
datacorr = data.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier','source'],axis=1)
plt.figure(figsize = (16,16))
sns.heatmap(datacorr.corr(),vmin=0,vmax=1,linewidths=0.5)

#######################################---MODELs---########################################
#Baseline Model
mean_sales = train['Item_Outlet_Sales'].mean()
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales
base1.to_csv('alg0.csv',index=False)

#############################Function for algo, predict, cv, score########################

target = 'Item_Outlet_Sales'
IDCol = ['Item_Identifier','Outlet_Identifier']

from sklearn import cross_validation, metrics

def modelfit(alg,dtrain,dtest,predictors,target,IDCol,filename):
    alg.fit(dtrain[predictors],dtrain[target])
    dtrain_predictions = alg.predict(dtrain[predictors])
    cv_score = cross_validation.cross_val_score(alg,dtrain[predictors],dtrain[target],cv=20,scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    print('\nModel Report\n')
    print('RMSE : %.4g' %np.sqrt(metrics.mean_squared_error(dtrain[target].values,dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    dtest[target]=alg.predict(dtest[predictors])
    #Export submission file:
    IDCol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDCol})
    submission.to_csv(filename, index=False)

#################################---End Function---#################################
    
#Linear Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDCol]
alg1 = LinearRegression(normalize=True)
modelfit(alg1,train,test,predictors,target,IDCol,'alg1.csv')
coef1 = pd.Series(alg1.coef_,predictors).sort_values()
plt.figure(figsize = (16,16))
coef1.plot(kind='bar',title='Model Coefficients')

#Ridge Regression
predictors = [x for x in train.columns if x not in [target]+IDCol]
alg2=Ridge(alpha=0.05,normalize=True)
modelfit(alg2,train,test,predictors,target,IDCol,'alg2.csv')
coef2 = pd.Series(alg2.coef_,predictors).sort_values()
plt.figure(figsize = (16,16))
coef2.plot(kind='bar',title='Model Coefficients')

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDCol]
alg3=DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg3,train,test,predictors,target,IDCol,'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_,predictors).sort_values(ascending=False)
plt.figure(figsize = (16,16))
coef3.plot(kind='bar',title='Feature Importances')

#Randon Forest
from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDCol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDCol, 'alg6.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
plt.figure(figsize = (16,16))
coef5.plot(kind='bar', title='Feature Importances')




# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:48:55 2017

@author: Jonathan
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset=pd.read_csv('heart_disease.csv')
X=dataset.iloc[:,:3].values
y=dataset.iloc[:,3].values

#backward elimination
# variables constant, age, cholesterol, heart rate
#answers which of these is important (multiple linear regression style)
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((270,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#from this we get heart rate being not significant, p value of 0.24
#remove heart rate as factor
X_opt=X[:,[0,1,2]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#now we get significant values, the proper set of variables for blood pressure
#mainly affected by age and cholesterol level

#let's look at AGE first
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,3].values
#Splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Heart Disease (Linear Regression)')
plt.xlabel('AGE')
plt.ylabel('Blood Pressure')
plt.show()

#Let's try the same thing with CHOLESTEROL
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,3].values
#Splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Heart Disease (Linear Regression)')
plt.xlabel('CHOL')
plt.ylabel('Blood Pressure')
plt.show()

#OK so now we have two SLR models. Maybe another model would work better.
#Let's try DECISION TREE model

#AGE
X=dataset.iloc[:,:1].values
y=dataset.iloc[:,3].values

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Heart Disease (Decision Tree Regression)')
plt.xlabel('AGE')
plt.ylabel('BLOOD PRESSURE')
plt.show()

#We can see that a more balanced graph appears where values increase more
#in the middle
# Predicting a new result
y_pred = regressor.predict(60)
# we get 133.92 as blood pressure, aged 60

#Cholesterol
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,3].values

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Heart Disease (Decision Tree Regression)')
plt.xlabel('Chol')
plt.ylabel('BLOOD PRESSURE')
plt.show()

#We can see that a zigzag graph is all over the place, not providing much help.
#Looks like in this case, the SLR models did just fine with helping us visualize.




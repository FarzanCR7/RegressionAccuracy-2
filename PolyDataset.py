# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 07:47:50 2021

@author: FARZAN
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
#Importing the DataSet
dataset=pd.read_csv("Poly_dataSet.csv")
x= dataset.iloc[:,0:1].values
y= dataset.iloc[:, 1].values
#%%
#Fitting Linear Regression to the DataSet
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
print(lin_reg.score(x,y)*100)
#%%
# Fitting the Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
print(lin_reg_2.score(x_poly,y)*100)
#%%
# Visualising the linear regression results
plt.scatter(x, y, color= "Black")
plt.plot(x, lin_reg.predict(x), color = "red")
plt.title("Truth or Bluff(Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
#%%
#Visualing the Polynomail regression result
plt.scatter(x, y, color= "Black")
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title("Truth or Bluff(Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
#%%
x_grid = np.arange(min(x), max(x), 0.1)
x_grid= x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color= "Black")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)))
plt.title("Truth or Bluff(Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
#%%
#Predicting a new result through Linear Regression
t=lin_reg.predict(np.reshape(6.5, (1,1)))
print(t)
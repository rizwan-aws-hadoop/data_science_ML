# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:26:29 2020

@author: M710583
"""

#import the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the data
df = pd.read_csv("C:/Users/M710583/Downloads/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv", sep=",",encoding="utf-8")
print(df)

#create matrix of features
X = df.iloc[:,:-1].values
print(X)
#create dpeendent variable vector
Y = df.iloc[:,1].values
print(Y)

#split data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

'''from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set results
Y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train, Y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#visualising the testng set results
plt.scatter(X_test, Y_test, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
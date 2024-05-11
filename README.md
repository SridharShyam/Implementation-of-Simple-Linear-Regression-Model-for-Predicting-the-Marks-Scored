# EXP-2: Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step - 1: 
Import the required libaries and read the dataframe.
### Step - 2: 
Assign hours to X and scores to Y.
### Step - 3: 
Implement the training set and the test set of the dataframe.
### Step - 4:
Plot the required graph for both the training data and the test data.
### Step - 5: 
Find the values of MSE,MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SHYAM S
RegisterNumber: 212223240156 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
data=pd.read_csv("C:/Users/admin/Documents/marks.csv")
X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print("Predicted Y value:",Y_pred)
print("Tested Y value:",Y_test)

plt.scatter(X_train,Y_train,color="darkcyan")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="grey")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
![image](https://github.com/SridharShyam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144871368/21b9ebf2-db29-424d-b986-7c16e06614bc)
![image](https://github.com/SridharShyam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144871368/c2b2584a-aa7e-465c-975d-2b5a5a29f3ba)
![image](https://github.com/SridharShyam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144871368/9d47fff8-fedf-4303-9c48-a7f9c0bced4a)
![image](https://github.com/SridharShyam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144871368/99bf2721-1649-43ee-9edc-eac26cf5ea48)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

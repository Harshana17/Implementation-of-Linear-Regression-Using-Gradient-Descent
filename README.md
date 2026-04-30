# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.

2.Scale the features using a standard scaler to normalize the data.

3.Initialize model parameters (theta) and add an intercept term to the feature set.

4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.

5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: HARSHANA M V
RegisterNumber: 212224240053
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones((len(X1), 1)), X1]   
    theta = np.zeros((X.shape[1], 1))    
    
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    
    return theta

data = pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")


X = data.iloc[:, :-1].drop(columns=['State']).values
y = data.iloc[:, -1].values.reshape(-1, 1)


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


theta = linear_regression(X_scaled, y_scaled)
print("Theta values:\n", theta)


new_data = np.array([[165349.2, 136897.8, 471784.1]])

new_scaled = scaler_X.transform(new_data)
new_scaled_bias = np.c_[np.ones((1, 1)), new_scaled]

prediction_scaled = new_scaled_bias.dot(theta)
prediction = scaler_y.inverse_transform(prediction_scaled)

print("\nPredicted scaled value:", prediction_scaled)
print("Predicted Profit:", prediction)
```
## Output:

<img width="1240" height="621" alt="image" src="https://github.com/user-attachments/assets/d0f956ae-f7d7-46c2-91bf-2d22a7fcf7d0" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

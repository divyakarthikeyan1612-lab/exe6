# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.The required Python libraries are imported and the placement dataset is loaded, after which the output column status is converted into numerical values.

2.The input features are selected and normalized using Standard Scaler, and a bias term is added to the dataset.

3.The sigmoid and cost functions are defined, initial weights are set, and gradient descent is applied to update the weights and reduce the cost.

4.Finally, predictions are made, accuracy is calculated, and a cost versus iterations graph is plotted. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("Placement_Data.csv")

# Convert Placed / Not Placed to 1 / 0
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Take only 2 features (simple)
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

# -----------------------------
# Standard Scaler (Normalization)
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias column (1)
m = len(y)
X = np.c_[np.ones(m), X]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

# Gradient Descent
theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

# Prediction
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

# -----------------------------
# PLOT: Cost vs Iterations
# -----------------------------
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
Developed by: DIVYA K
RegisterNumber:  25019198
*/
```

## Output:
![logistic regression using gradient descent]
<img width="792" height="644" alt="image" src="https://github.com/user-attachments/assets/69748969-74b9-40ec-8143-db33cacfab71" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


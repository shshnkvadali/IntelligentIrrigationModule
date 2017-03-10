import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

ifile  = open('bills_updated.csv', "r")
read = list(csv.reader(ifile))
read.pop(0)
input,output=[],[]
for row in read:
    input.append(row[:-1])
    output.append(row[-1:])
input=np.array(input).astype(np.float)
output=np.array(output).astype(np.float)
input=np.array(input)
output=np.array(output)

diabetes_X_train = input[:-315]
diabetes_X_test = input[-315:]
# Split the targets into training/testing sets
diabetes_y_train = output[:-315]
diabetes_y_test = output[-315:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))


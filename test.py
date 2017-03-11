import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import datasets, linear_model
from linearRegression import generateModel
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

diabetes_X_train = input[:-134]
diabetes_X_test = input[-134:]
# Split the targets into training/testing sets
diabetes_y_train = output[:-134]
diabetes_y_test = output[-134:]

regr = generateModel(diabetes_X_train,diabetes_y_train, "linear")
print('Linear Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

regr = generateModel(diabetes_X_train,diabetes_y_train, "ridge")

print('Ridge Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

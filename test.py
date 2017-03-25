import matplotlib.pyplot as plt
import csv
import numpy as np
from linearRegression import generateModel
from sklearn.cross_validation import train_test_split
ifile  = open('dataset.csv', "r")
read = list(csv.reader(ifile, delimiter=','))
read.pop(0)
input,output=[],[]
for row in read:
    #input.append(row[4:6]+row[7:9]+row[11:13])
    input.append(row[6:14])
    output.append(row[15])
print('Input',input[0])
print('Output',output[0])
input=np.array(input).astype(np.float)
output=np.array(output).astype(np.float)
input=np.array(input)
output=np.array(output)

X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)

regr = generateModel(X_train,y_train, "linear")
print('Linear Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The coefficients
#print('Residues: \n', regr.residues_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

regr = generateModel(X_train,y_train, "ridge")

print('Ridge Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

regr = generateModel(X_train,y_train, "lasso")
print('Lasso Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

regr = generateModel(X_train,y_train, "theilSen")
print('TheilSen Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

regr = generateModel(X_train,y_train, "ElasticNet")
print('ElasticNet Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

regr = generateModel(X_train,y_train, "bayesianRidge")
print('BayesianRidge Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

regr = generateModel(X_train,y_train, "RANSACRegressor")
print('RANSACRegressor Regression \n')
# The coefficients
print('Coefficients: \n', regr.estimator_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

print('Size x: %i  and y: %i',X_test.shape,y_test.shape)
# Plot outputs

#plt.scatter(X_test[:,0:3], y_test,  color='black')
#plt.plot(X_test, regr.predict(X_test), color='blue',
#        linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()

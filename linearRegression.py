import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

def generateModel(x_train, y_train, type):
	# Create linear regression object
	if type == 'linear' :
		regr = linear_model.LinearRegression()
	if type == "ridge" :
		regr = linear_model.Ridge(alpha = .5)
	elif type == "lasso" :
		regr = linear_model.Lasso(alpha = .1)
	elif type == "lassoLars" :
		regr = linear_model.LassoLars(alpha = .1)
	# Train the model using the training sets
	regr.fit(x_train, y_train)

	return regr
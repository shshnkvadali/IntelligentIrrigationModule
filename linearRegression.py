
from sklearn import linear_model

def generateModel(x_train, y_train, type):
	# Create linear regression object
    
    if type == 'linear' :
        regr = linear_model.LinearRegression()
    elif type == "ridge" :
        regr = linear_model.Ridge(alpha = .1)
    elif type == "lasso" :
        regr = linear_model.Lasso(alpha = .1)
    elif type == "bayesianRidge" :
        regr = linear_model.BayesianRidge()
    elif type == "RANSACRegressor" :
        regr = linear_model.RANSACRegressor(random_state=42)
    elif type == "theilSen" :
        regr = linear_model.TheilSenRegressor(random_state=42)
    elif type == "ElasticNet" :
        regr = linear_model.ElasticNet(l1_ratio=0.7)
        
	# Train the model using the training sets
    regr.fit(x_train, y_train)

    return regr
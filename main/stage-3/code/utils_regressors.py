import numpy as np
from utils import all_x, all_y
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import GridSearchCV
seed = 42
beta_grid = np.concatenate((np.linspace(.1, .9, 9), np.linspace(1, 10, 10)))
depth_grid = range(2, 11)

###################################
## K-nearest Neighbours


def KNNReg(data, n_neighbors_grid=range(1, 11)):
    train_x, test_x, train_y, test_y = data

    parameters = {'n_neighbors':n_neighbors_grid}

    from sklearn.neighbors import KNeighborsRegressor
    knnr = KNeighborsRegressor(n_jobs=-1)
    clf = GridSearchCV(knnr, parameters, cv=5,
                       scoring='neg_mean_squared_log_error',
                       n_jobs=-1)

    clf.fit(train_x, train_y)

    test_y_hat = clf.predict(test_x)
    mse = mean_squared_log_error(test_y, test_y_hat)
    print(mse)
    return clf, mse

###################################
## Linear Regression


def LinReg(data):
    train_x, test_x, train_y, test_y = data

    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression(n_jobs=-1).fit(train_x, train_y)

    test_y_hat = linreg.predict(test_x)
    mse = mean_squared_log_error(test_y, test_y_hat)
    print(mse)
    return linreg, mse


###################################
## SVM


def SVR(data, beta_grid=beta_grid):
    train_x, test_x, train_y, test_y = data

    parameters = {'kernel':('linear', 'rbf'), 'C':beta_grid}

    from sklearn.svm import SVR
    svr = SVR()
    clf = GridSearchCV(svr, parameters, cv=5,
                       scoring='neg_mean_squared_log_error',
                       n_jobs=-1)
    clf.fit(train_x, train_y)

    test_y_hat = clf.predict(test_x)
    mse = mean_squared_log_error(test_y, test_y_hat)
    print(mse)
    return clf, mse


###################################
## Random Forest

def RFR(data, seed=seed, depth_grid=depth_grid):
    train_x, test_x, train_y, test_y = data

    parameters = {'max_depth':depth_grid}

    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(n_jobs=-1)
    clf = GridSearchCV(rfr, parameters, cv=5,
                       scoring='neg_mean_squared_log_error',
                       n_jobs=-1)

    clf.fit(train_x, train_y)

    test_y_hat = clf.predict(test_x)
    mse = mean_squared_log_error(test_y, test_y_hat)
    print(mse)
    return clf, mse


###################################
## Bagging

def BagReg(data, seed=seed, depth_grid=depth_grid):
    train_x, test_x, train_y, test_y = data

    parameters = {'max_depth':depth_grid}

    from sklearn.ensemble import BaggingRegressor
    bag = BaggingRegressor(n_estimators=1000,
                           random_state=seed, n_jobs=-1)

    bag.fit(train_x, train_y)

    test_y_hat = bag.predict(test_x)
    mse = mean_squared_log_error(test_y, test_y_hat)
    print(mse)
    return bag, mse

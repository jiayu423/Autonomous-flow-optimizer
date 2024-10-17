from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def peakarea(d):
    score = d[:, -1]
    temp = d[:, 0]
    rt = d[:, 1]
    ester_eq = d[:, 2]
    ammonia_eq = d[:, 3]

    return 624 * (score - ((1 / 22) * (140 - temp) + (5 / 4) * (6 - ester_eq) + (5 / 18) * (20 - rt) + (5 / 12) * (
                16 - ammonia_eq))) / 80

def model():

    print('training GP models for peak area simulations......')
    data_multi = pd.read_csv('C:/Users/User/Documents/PyCharm Projects/flow_edbo_integration/utils/HAN-10-Luca.csv').to_numpy()
    data_single = pd.read_csv('C:/Users/User/Documents/PyCharm Projects/flow_edbo_integration/utils/HAN-9-edbo.csv').to_numpy()
    data_single[:, -1] = peakarea(data_single)
    data = np.vstack((data_single, data_multi))

    X = data[:, :-1]
    y = data[:, -1]

    X_ = np.zeros(X.shape)
    min_list, max_list = [], []
    for i in range(4):
        X_[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
        min_list.append(X[:, i].min())
        max_list.append(X[:, i].max())

    GP = GaussianProcessRegressor(alpha=0.1, kernel=Matern(length_scale=0.5, nu=0.5),
                         normalize_y=True)
    GP.fit(X_, y)

    # parameters = {'alpha': np.linspace(0.1, 1, 20),
    #               'kernel': [Matern(i, nu=0.5) for i in np.linspace(0.1, 1, 10)]}
    # regr = GaussianProcessRegressor(normalize_y=True)
    # clf = GridSearchCV(regr, parameters)
    # clf.fit(X_, y)

    # print(clf.best_estimator_)
    #
    # res = clf.predict(X_)
    # plt.scatter(res, y)
    # plt.show()

    return GP, min_list, max_list

# m, m1, m2 = model()

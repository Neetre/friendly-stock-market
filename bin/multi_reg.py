'''
Let's try with multiple regression instead of RNN

Neetre 2024
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import seaborn as sns
sns.set_theme()


def load_data(stock):
    data = pd.read_csv(f'../data/csv_preprocessed/{stock}.csv')
    print(data.describe())
    # print(data["Date"])
    data = data.drop("Date", axis=1)
    X = data.drop('Close', axis=1)
    y = data['Close']
    return X, y


def adj_r2(reg, X, y):
    r2 = reg.score(X, y)
    n = X.shape[0]  # Number of observations
    p = X.shape[1]  # Number of features/predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2


def analyze(stock=""):
    X, y = load_data(stock)
    reg = LinearRegression()
    reg.fit(X, y)
    print("Regression coeff: ", reg.coef_)
    print("Regression r2 score: ", reg.score(X, y))
    print("Regression adjusted r2 score: ", adj_r2(reg, X, y))

    p_values = f_regression(X, y)[1]
    p_values = p_values.round(3)
    print("P-values: ", p_values)

    # print(reg.predict([[124.09,126.41,122.92,434800000]]))  # 2024-06-27 00:00:00-04:00,124.09,126.41,122.92,,434800000


if __name__ == "__main__":
    analyze("GOOGL")

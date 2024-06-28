'''
Let's try with multiple regression instead of RNN

Neetre 2024
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set_theme()



def load_data():
    data = pd.read_csv('../data/csv_preprocessed/AMD.csv')
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


def main():
    X, y = load_data()
    reg = LinearRegression()
    reg.fit(X, y)
    print("Regression coeff: ", reg.coef_)
    print("Regression r2 score: ", reg.score(X, y))
    print("Regression adjusted r2 score: ", adj_r2(reg, X, y))

    # print(reg.predict([[124.00,126.41,122.92,434800000]]))  # 2024-06-27 00:00:00-04:00,124.09,126.41,122.92,,434800000


if __name__ == "__main__":
    main()

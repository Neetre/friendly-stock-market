'''
Let's try with multiple regression instead of RNN

Neetre 2024
'''

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

from web_scraping import webscarp, convert_to_float


def load_data(key, crypto):
    try:
        data = pd.read_csv(f'../data/csv_preprocessed/' + ('crypto' if crypto else 'stock') + f'/{key}.csv')
        data = data.drop("Date", axis=1)
        X = data.drop('Close', axis=1)
        y = data['Close']
        return X, y
    except FileNotFoundError:
        print(f"File not found for {key}. Please ensure the file exists and is in the correct directory.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading data for {key}: {e}")
        return None, None

def adj_r2(reg, X, y):
    r2 = reg.score(X, y)
    n = X.shape[0]  # Number of observations
    p = X.shape[1]  # Number of features/predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2

def analyze(key="", crypto=False):
    print(f"Analyzing {key}")
    X, y = load_data(key, crypto)
    if X is None or y is None:
        print("Data loading failed. Analysis aborted.")
        return
    try:
        reg = LinearRegression()
        reg.fit(X, y)
        print("Regression coeff: ", reg.coef_)
        print("Regression r2 score: ", reg.score(X, y))
        print("Regression adjusted r2 score: ", adj_r2(reg, X, y))

        p_values = f_regression(X, y)[1]
        p_values = p_values.round(3)
        print("P-values: ", p_values)
        return reg

    except ValueError as e:
        print(f"Error during regression analysis: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during analysis of {key}: {e}")


def predicts(key, reg, crypto=False, auto=False):
    print(f"Predicting for {key}...")
    if auto:
        data = webscarp(key, crypto)
    
        prediction = reg.predict([[data['Prev_close'], data['High'], data['Low'], data['Volume']]])  # [float(i) for i in data.split(",")]
    else:
        data = input("Enter the data in the format 'open_price,high,low,volume': ")
        prediction = reg.predict([[float(i) if 'M' not in i else convert_to_float(i) for i in data.split(",")]])
        
    print(f"Closing price predicted for {key}: {prediction[0]:.4f}$")


if __name__ == "__main__":
    analyze("BTC-USD", True)  # 2024-06-28 00:00:00+00:00,61612.8046875,62126.09765625,61190.26171875,61439.4375,22124865536

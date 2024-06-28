'''
This script is used to preprocess the data for the model training.

Neetre 2024
'''

import os
import pandas as pd
import numpy as np


def preprocess_data(crypto):
    '''
    Structure of csv files: Date,Open,High,Low,Close,Volume,Dividends,Stock Splits,Name
    We want Date,Open,High,Low,Close,Volume
    '''

    # Get the list of csv files
    if crypto:
        csv_dir = os.path.join('..', 'data', 'csv', 'crypto')
    else:
        csv_dir = os.path.join('..', 'data', 'csv', 'stock')

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    # Create a new directory to store the preprocessed data
    if crypto:
        preprocessed_data_dir = os.path.join('..', 'data', 'csv_preprocessed', 'crypto')
    else:
        preprocessed_data_dir = os.path.join('..', 'data', 'csv_preprocessed', 'stock')

    os.makedirs(preprocessed_data_dir, exist_ok=True)

    for csv_file in csv_files:
        print(f"Preprocessing {csv_file}")
        df = pd.read_csv(os.path.join(csv_dir, csv_file))
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(os.path.join(preprocessed_data_dir, csv_file), index=False)

    return


if __name__ == '__main__':
    preprocess_data(True)
    preprocess_data(False)
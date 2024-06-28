import os
import pandas as pd

def preprocess_data(crypto):
    '''
    Structure of csv files: Date,Open,High,Low,Close,Volume,Dividends,Stock Splits,Name
    We want Date,Open,High,Low,Close,Volume
    '''

    try:
        # Get the list of csv files
        if crypto:
            csv_dir = os.path.join('..', 'data', 'csv', 'crypto')
        else:
            csv_dir = os.path.join('..', 'data', 'csv', 'stock')

        if not os.path.exists(csv_dir):
            print(f"Directory does not exist: {csv_dir}")
            os.makedirs(csv_dir, exist_ok=True)
            return

        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

        # Create a new directory to store the preprocessed data
        if crypto:
            preprocessed_data_dir = os.path.join('..', 'data', 'csv_preprocessed', 'crypto')
        else:
            preprocessed_data_dir = os.path.join('..', 'data', 'csv_preprocessed', 'stock')

        os.makedirs(preprocessed_data_dir, exist_ok=True)

        for csv_file in csv_files:
            try:
                print(f"Preprocessing {csv_file}")
                df = pd.read_csv(os.path.join(csv_dir, csv_file))
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df.to_csv(os.path.join(preprocessed_data_dir, csv_file), index=False)
            except pd.errors.EmptyDataError:
                print(f"No data in file: {csv_file}")
            except pd.errors.ParserError:
                print(f"Error parsing file: {csv_file}")
            except Exception as e:
                print(f"An error occurred while processing {csv_file}: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    preprocess_data(True)
    preprocess_data(False)
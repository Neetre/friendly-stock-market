'''
This script downloads the stock data from Yahoo Finance for the S&P 500 companies

Neetre 2024
'''

import os

from datetime import datetime
from concurrent import futures

import ccxt
import yfinance as yf
import time

import pandas as pd


bad_names =[] #to keep track of failed queries
bad_names_crypto =[] #to keep track of failed queriess


def download_crypto(crypto):
    try:
        # Fetch data for the cryptocurrency
        data = yf.Ticker(crypto)
        # Get historical market data
        crypto_df = data.history(period="max")
        if crypto_df.empty:
            raise ValueError(f"No data for {crypto}")
        crypto_df['Name'] = crypto
        output_file = os.path.join('..', 'data', 'csv', 'crypto', f'{crypto}.csv')
        crypto_df.to_csv(output_file)
        time.sleep(1)
    except Exception as e:
        print(f"Failed to download {crypto}: {e}")
        bad_names_crypto.append(crypto)
        return None


def download_stock(stock):
    '''
    This function downloads the stock data from yahoo finance
    and saves it to a csv file
    '''
    global bad_names

    try:
        print(f"Downloading {stock}")
        stock_data = yf.Ticker(stock)
        stock_df = stock_data.history(period='max')
        if stock_df.empty:
            raise ValueError(f"No data for {stock}")
        stock_df['Name'] = stock
        output_file = os.path.join('..', 'data', 'csv', 'stock', f'{stock}.csv')
        stock_df.to_csv(output_file)
        time.sleep(1)
    except Exception as e:
        bad_names.append(stock)
        print(f"Error downloading {stock}: {str(e)}")


def get_s_and_p500():
    '''This function downloads the S&P 500 companies from Wikipedia'''
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]

    return df['Symbol'].to_list()


def get_crypto_list():
    exchange = ccxt.binance()
    exchange.load_markets()
    crypto_list = list(exchange.markets.keys())
    crypto_list = [i for i in crypto_list if 'USD' == i.split("/")[1].split(":")[0]]
    crypto_list = [i.replace('/', "-").split(":")[0] for i in crypto_list]
    crypto_list = list(set(crypto_list))
    crypto_list = sorted(crypto_list)
    return crypto_list


def get_data_stock():
    global bad_names

    now_time = datetime.now()

    s_and_p = get_s_and_p500()
    max_workers = 10
    s_and_p_sorted = sorted(s_and_p)
    workers = min(max_workers, len(s_and_p_sorted))
    with futures.ThreadPoolExecutor(workers) as executor:
        res = executor.map(download_stock, s_and_p_sorted)

    if len(bad_names) > 0:
        with open('failed_queries.txt','w') as outfile:
            for name in bad_names:
                outfile.write(name+'\n')

    finish_time = datetime.now()
    duration = finish_time - now_time
    minutes, seconds = divmod(duration.seconds, 60)
    print('get_data_threaded.py')
    print(f"The threaded script took {minutes} minutes and {seconds} seconds to run.")


def get_data_crypto():
    global bad_names_crypto

    now_time = datetime.now()
    crypto_list = get_crypto_list()
    max_workers = 10
    crypto_list_sorted = sorted(crypto_list)
    workers = min(max_workers, len(crypto_list_sorted))
    with futures.ThreadPoolExecutor(workers) as executor:
        res = executor.map(download_crypto, crypto_list_sorted)

    if len(bad_names_crypto) > 0:
        with open('failed_queries_crypto.txt','w') as outfile:
            for name in bad_names_crypto:
                outfile.write(name+'\n')

    finish_time = datetime.now()
    duration = finish_time - now_time
    minutes, seconds = divmod(duration.seconds, 60)
    print('get_data_threaded.py')
    print(f"The threaded script took {minutes} minutes and {seconds} seconds to run.")


def main():
    get_data_crypto()
    # get_data_stock()

if __name__ == '__main__':
    main()
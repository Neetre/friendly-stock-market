'''
This is the main file that will be used to run the program.
It will take in arguments from the command line and run the program accordingly.

Neetre 2024
'''


from get_data_threaded import get_data_stock, get_data_crypto
from multi_reg import analyze, predicts
from preprocess_data import preprocess_data
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description="Analyze stock data using multiple regression")
    parser.add_argument("--download", action="store_true", help="Download stock data")
    parser.add_argument("--crypto",  action="store_true", help="Predict for cryptos")
    parser.add_argument("--stock", action="store_true", help="Predict for stock")
    parser.add_argument("--predict", action="store_true", help="Predict for the chosen stock or crypto")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    return parser.parse_args()


def main():
    args = arg_parser()

    if args.crypto:
        if args.download:
            get_data_crypto()
            preprocess_data(True)

        crypto = input("Enter the cryptocurrency you want to analyze: ")

        reg = analyze(crypto, True)

        if args.predict:
            predicts(crypto, reg, True)


    else:
        if args.download:
            get_data_stock()
            preprocess_data(False)

        stock = input("Enter the stock you want to analyze(acronym): ")

        reg = analyze(stock, False)

        if args.predict:
            predicts(stock, reg, False)


if __name__ == "__main__":
    main()
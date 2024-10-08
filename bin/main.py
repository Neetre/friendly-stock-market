'''
This is the main file that will be used to run the program.
It will take in arguments from the command line and run the program accordingly.

Neetre 2024
'''


from get_data_threaded import get_data_stock, get_data_crypto
from multi_reg import analyze, predicts
from preprocess_data import preprocess_data
import argparse

from icecream import ic


def arg_parser():
    parser = argparse.ArgumentParser(description="Analyze stock data using multiple regression")
    parser.add_argument("--download", action="store_true", help="Download stock data")
    parser.add_argument("--crypto",  action="store_true", help="Predict for cryptos")
    parser.add_argument("--stock", action="store_true", help="Predict for stock")
    parser.add_argument("--predict", action="store_true", help="Predict for the chosen stock or crypto")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    return parser.parse_args()


def main():
    try:
        args = arg_parser()
        
        if args.verbose:
            ic.enable()
        else:
            ic.disable()

        if args.crypto:
            if args.download:
                get_data_crypto()
                preprocess_data(True)

            crypto = input("Enter the cryptocurrency you want to analyze(BTC, ETH, ...): ") + "-USD"

            try:
                reg = analyze(crypto, True)
            except Exception as e:
                print(f"Error analyzing {crypto}: {e}")
                return

            if args.predict:
                try:
                    predicts(crypto, reg, True)
                except Exception as e:
                    print(f"Error predicting {crypto}: {e}")
                    return

        else:
            if args.download:
                get_data_stock()
                preprocess_data(False)

            stock = input("Enter the stock you want to analyze(acronym): ")

            try:
                reg = analyze(stock, False)
            except Exception as e:
                print(f"Error analyzing {stock}: {e}")
                return

            if args.predict:
                try:
                    predicts(stock, reg, False)
                except Exception as e:
                    print(f"Error predicting {stock}: {e}")
                    return

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
from get_data_threaded import get_data
from multi_reg import analyze
from preprocess_data import preprocess_data
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description="Analyze stock data using multiple regression")
    parser.add_argument("--download", action="store_true", help="Download stock data")
    parser.add_argument("--stock", type=str, default="GOOGL",  help="Stock symbol to analyze (default: GOOGL)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    return parser.parse_args()


def main():
    args = arg_parser()
    if args.download:
        get_data()
        preprocess_data()

    analyze(args.stock)


if __name__ == "__main__":
    main()
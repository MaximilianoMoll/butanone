import argparse
from Anonymizer import Anonymizer

parser = argparse.ArgumentParser("K-Anonymize")
parser.add_argument("--method", type=str, default="mondrian", help="K-Anonymity Method")
parser.add_argument("--k", type=int, default=2, help="K-Anonymity or L-Diversity")
parser.add_argument("--dataset", type=str, default="adult", help="Dataset to anonymize")


def main(args):
    anonymizer = Anonymizer(args)
    anonymizer.anonymize()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

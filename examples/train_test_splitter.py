from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE_DEFAULT = 0.2
RANDOM_SEED_DEFAULT = 42


def split_dataset(
    in_file: str,
    out_file_train: str,
    out_file_test: str,
    test_size: float,
    random_seed: int,
):
    df = pd.read_csv(in_file)

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_seed
    )

    df_train.to_csv(out_file_train, index=None)
    df_test.to_csv(out_file_test, index=None)


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE_DEFAULT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    split_dataset(args.file, args.train, args.test, args.test_size, args.random_seed)

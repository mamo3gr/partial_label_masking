from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

genre_all = [
    "action",
    "adventure",
    "animation",
    "comedy",
    "crime",
    "drama",
    "fantasy",
    "horror",
    "mystery",
    "romance",
    "sci-fi",
    "short",
    "thriller",
]


def load_dataset(root_dir: Union[str, Path]):
    root_dir = Path(root_dir)
    csv_path = root_dir / "duplicate_free_41K.csv"
    image_dir = root_dir / "images"

    df = pd.read_csv(csv_path)

    df["multi_hot"] = df[genre_all].values.tolist()
    df["multi_hot"] = df["multi_hot"].apply(np.array)

    df["file"] = df["id"].apply(lambda x: str(image_dir / Path(f"{x}.jpg")))

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    return (
        df_train["file"].tolist(),
        df_test["file"].tolist(),
        df_train["multi_hot"].tolist(),
        df_test["multi_hot"].tolist(),
    )


if __name__ == "__main__":
    root_dir = Path("/home/mamo/datasets/img_41K")
    load_dataset(root_dir)

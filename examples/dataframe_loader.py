from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class DataFrameLoader:
    def __init__(
        self,
        filename_col: str,
        labels: List[str],
        image_dir: Optional[Union[str, Path]] = None,
        filename_postfix: str = "",
        logger=None,
    ):
        if logger is None:
            logger = getLogger(__name__)
        self._logger = logger

        if image_dir is None:
            image_dir = Path("./")
            self._logger.info(
                "Image directory not specified. Set it as the current directory"
            )

        self.filename_col = filename_col
        self.filename_postfix = filename_postfix
        self.labels = labels
        self.image_dir = image_dir

    def load(self, csv_path=Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path)
        self._logger.info(f"Load {csv_path}")

        def fullpath(base, directory, postfix) -> str:
            basename = Path(f"{base}{postfix}")
            fpath = Path(directory) / basename
            return str(fpath)

        paths = df[self.filename_col].apply(
            lambda base: fullpath(base, self.image_dir, self.filename_postfix)
        )
        y = np.array(df[self.labels], np.int64)
        self._logger.info(f"The csv file contains {len(paths)} images")
        self._logger.info(f"Number of labels: {len(self.labels)} {self.labels}")

        return paths, y

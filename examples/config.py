import dataclasses
from pathlib import Path
from typing import List, Union

import yaml


@dataclasses.dataclass
class Config:
    csv_path: str
    image_dir: str
    filename_col: str
    labels: List[str]
    image_height: int
    image_width: int
    batch_size: int
    random_seed: int
    validation_ratio: float
    weight_decay: float
    epochs: int
    model_path: str
    save_best_only: bool = True
    filename_postfix: str = ""


def load_config(yaml_path: Union[str, Path]) -> Config:
    with open(yaml_path) as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)
    return Config(**config_dict)

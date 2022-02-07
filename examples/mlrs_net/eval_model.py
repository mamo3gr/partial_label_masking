import json
from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from examples.config import Config, load_config
from examples.dataframe_loader import DataFrameLoader
from examples.dataset_generator import DatasetGenerator
from examples.model import create_model
from examples.utils import get_y_true, set_gpu_memory_growth

logger = getLogger(__name__)


def test_model(conf: Config, output_file):
    set_gpu_memory_growth()

    test_ds = _create_test_set(conf)
    model = _create_and_load_model(conf)

    proba = model.predict(test_ds, verbose=1)
    y_pred = np.where(proba >= 0.5, 1, 0)
    y_true = get_y_true(test_ds)
    labels = conf.labels
    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )

    with open(output_file, "w") as f:
        json.dump(report, f)


def _create_test_set(conf: Config):
    loader = DataFrameLoader(
        filename_col=conf.filename_col, labels=conf.labels, image_dir=conf.image_dir
    )
    paths, y = loader.load(conf.csv_path)

    gen = DatasetGenerator(
        image_height=conf.image_height,
        image_width=conf.image_width,
        batch_size=conf.batch_size,
        preprocess_func=tf.keras.applications.resnet_v2.preprocess_input,
        logger=logger,
    )
    ds = gen.generate(paths, y)

    return ds


def _create_and_load_model(conf: Config):
    labels = conf.labels
    n_classes = len(labels)
    input_shape = (conf.image_height, conf.image_width, 3)
    model = create_model(
        input_shape=input_shape,
        n_classes=n_classes,
        weights_decay=conf.weight_decay,
        backbone_class=tf.keras.applications.ResNet50V2,
        use_pretrain=True,
    )
    model.load_weights(conf.model_path)
    return model


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("-o", "--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    basicConfig(level=INFO)
    args = _parse_args()
    config = load_config(args.config_file)
    test_model(config, args.output)

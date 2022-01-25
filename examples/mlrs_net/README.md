# Multi-label classification with MLRSNet dataset
## Reference
* [cugbrs/MLRSNet: A Multi-label High Spatial Resolution Remote Sensing Dataset for Semantic Scene Understanding](https://github.com/cugbrs/MLRSNet)

## Setup
```shell
DATASET_DIR=/path/to/arbitrary/directory
# clone repo
git clone git@github.com:cugbrs/MLRSNet.git $DATASET_DIR
# decompress archived images
./unrar_all.sh $DATASET_DIR/Images
# merge all csv files
./merge_csv_files.sh $DATASET_DIR/labels $DATASET_DIR/labels.csv
# split the csv file into train/test
poetry run python ../train_test_splitter.py $DATASET_DIR/labels.csv \
--test-size=0.7 \
--random-seed=42 \
--train $DATASET_DIR/train30.csv \
--test $DATASET_DIR/test70.csv
```

## Train model
```shell
PYTHONPATH=../../ poetry run python train_model.py train.yaml
```

You can see training log with Tensorboard:

```shell
tensorboard --logdir ./logs
```

## Evaluate model
```shell
PYTHONPATH=../../ poetry run python eval_model.py eval.yaml
```

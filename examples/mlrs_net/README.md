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

## Example run and result

```shell
cd ../..  # go to repository root

# check commit hash
git rev-parse HEAD
3e5fd2e195b0b67f08e6aa19c86788cb5ef5674e

# train model without/with PLM
PYTHONPATH=$(pwd) poetry run python examples/mlrs_net/train_model.py examples/mlrs_net/train.yaml
PYTHONPATH=$(pwd) poetry run python examples/mlrs_net/train_model_plm.py examples/mlrs_net/train_plm.yaml

# evaluate each model
PYTHONPATH=$(pwd) poetry run python examples/mlrs_net/eval_model.py examples/mlrs_net/eval.yaml -o examples/mlrs_net/naive.json
PYTHONPATH=$(pwd) poetry run python examples/mlrs_net/eval_model.py examples/mlrs_net/eval_plm.yaml -o examples/mlrs_net/plm.json

# open and run compare_metrics.ipynb
PYTHONPATH=$(pwd) poetry run jupyter notebook
```

# partial_label_masking
TensorFlow implementation of Partial Label Masking (PLM)

## Prerequisite
* [Poetry](https://python-poetry.org/) >= 1.1.12
* GNU Make

## Installation
```shell
make dep
```

## Example
See [examples/mlrs_net/train_model_plm.py](examples/mlrs_net/train_model_plm.py)

## Format, lint and test
```shell
make format
make lint
make test
```

## Reference
K. Duarte, Y. Rawat and M. Shah, "[PLM: Partial Label Masking for Imbalanced Multi-Label Classification](https://openaccess.thecvf.com/content/CVPR2021W/LLID/html/Duarte_PLM_Partial_Label_Masking_for_Imbalanced_Multi-Label_Classification_CVPRW_2021_paper.html)," CVPR2021 Workshops, pp. 2739-2748.

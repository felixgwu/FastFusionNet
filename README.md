# FastFusionNet

## Overview
This repo contains the code of [_FastFusionNet: New State-of-the-Art for DAWNBench SQuAD_](https://arxiv.org/abs/1902.11291v1).


## Requirements
```
torch==0.3.1
spacy==1.9.0
numpy
pandas
tqdm
tesnorboardX
```
Please also install the SRU version 1 from [here](https://github.com/felixgwu/sru).
Please download GloVe (Pennington et al., EMNLP 2014) and CoVe (McCann et al., NIPS 2017) by
```sh
bash download.sh
```

## Preprocessing
Preprocessing the data set. This takes about 10 minutes.
`PATH_TO_SQAUD_TRAIN` should be the path to `train-v1.1.json` and `PATH_TO_SQAUD_DEV` should be the path to `dev-v1.1.json`. This will generate the preprocessed data file at `data/squad/data-fusion.pth`.
```sh
mkdir -p data/squad
python prepro.py --train PATH_TO_SQAUD_TRAIN --dev PATH_TO_SQUAD_DEV
```

## Training
To train FastFusionNet:
```sh
SAVE='save/fastfusionnet'
mkdir -p $SAVE
python train.py --model_type fusionnet --hidden_size 125 --end_gru \
    --dropout_rnn 0.2 --data_suffix fusion --save_dir $SAVE \
    -lr 0.001 -gc 20  -e 100 --batch_size 32 \
    --rnn_type sru --fusion_reading_layers 2 --fusion_understanding_layers 2 --fusion_final_layers 2
```

To train FusionNet:
```sh
SAVE='save/fusionnet'
mkdir -p $SAVE
python train.py --model_type fusionnet --hidden_size 125 --end_gru \
    --dropout_rnn 0.2 --data_suffix fusion --save_dir $SAVE \
    -lr 0.001 -gc 20  -e 100 --batch_size 32 \
    --rnn_type lstm --fusion_reading_layers 1 --fusion_understanding_layers 1 --fusion_final_layers 1 --use_CoVe
```

## Reference
```
@article{wu2019fastfusionnet,
  title={FastFusionNet: New State-of-the-Art for DAWNBench SQuAD},
  author={Wu, Felix and Li, Boyi and Wang, Lequn and Lao, Ni and Blitzer, John and and Weinberger, Kilian Q.},
  journal={arXiv preprint arXiv:1902.11291},
  url={https://arxiv.org/abs/1902.11291},
  year={2019}
}
```

# FastFusionNet

## Overview
This repo contains the code of [_FastFusionNet: New State-of-the-Art for DAWNBench SQuAD_](https://arxiv.org/abs/1902.11291).

## News
We now support PyTorch version `>=0.4.1` in a new [branch](https://github.com/felixgwu/FastFusionNet/tree/v1.0.1). However, it is slightly slower.

## Requirements
```
torch==0.3.1
spacy==1.9.0
numpy
pandas
tqdm
tesnorboardX
oldsru
```
Please also install the SRU version 1 (`oldsru`) from [here](https://github.com/felixgwu/oldsru).
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
To train FastFusionNet [(Wu et al., arXiv 2019)](https://arxiv.org/abs/1902.11291v1):
```sh
SAVE='save/fastfusionnet'
mkdir -p $SAVE
python train.py --model_type fusionnet --hidden_size 125 --end_gru \
    --dropout_rnn 0.2 --data_suffix fusion --save_dir $SAVE \
    -lr 0.001 -gc 20  -e 100 --batch_size 32 \
    --rnn_type sru --fusion_reading_layers 2 --fusion_understanding_layers 2 --fusion_final_layers 2
```

To train FusionNet [(Huang et al., ICLR 2018)](https://arxiv.org/abs/1711.07341):
```sh
SAVE='save/fusionnet'
mkdir -p $SAVE
python train.py --model_type fusionnet --hidden_size 125 --end_gru \
    --dropout_rnn 0.4 --data_suffix fusion --save_dir $SAVE \
    -lr 0.001 -gc 20  -e 100 --batch_size 32 \
    --rnn_type lstm --fusion_reading_layers 1 --fusion_understanding_layers 1 --fusion_final_layers 1 --use_cove
```

To train GLDR-DrQA [(Wu et al., arXiv 2017)](https://arxiv.org/abs/1711.04352):
```sh
python train.py --model_type gldr-drqa --hidden_size 128 \
    --dropout_rnn 0.2 --data_suffix fusion --save_dir $SAVE \
    -lr 0.001 -gc 20  -e 100 --batch_size 32 \
    -doc_layers 17 --question_layers 9
```

## Evalutation
To evaluate the best trained model in 'save/fastfusionnet' and get the latency (batch size=1):
```sh
python eval.py --save_dir save/fastfusionnet --resume best_model.pt --eval_batch_size 1
```

## Pre-trained model
FastFusionNet model [link](https://cornell.box.com/s/7cunr95j9xcigo9wa05jvike8my7z3o5) dev EM: 73.58 F1: 82.42

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
## Acknowledgement
This is based on the v0.3.1 version of Runqi Yang's excellent [DrQA code base](https://github.com/hitvoice/DrQA/tree/4ad445276373173d7f5845352a4fff910bf1239e) as well as the official [FusionNet on NLI](https://github.com/momohuang/FusionNet-NLI) implementation.
Lots of Runqi's code is borrowed from [Facebook/ParlAI](https://github.com/facebookresearch/ParlAI/) under an MIT license.

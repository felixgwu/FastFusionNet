#!/usr/bin/env bash

# Download GloVe
mkdir -p data/glove
if [ ! -f data/glove/glove.840B.300d.txt ]; then
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove/glove.840B.300d.zip
    unzip data/glove/glove.840B.300d.zip -d data/glove
fi

# Download CoVe
if [ ! -f data/glove/MT-LSTM.pth ]; then
    wget https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth -O data/glove/MT-LSTM.pth
fi


# Download SpaCy English language models
python -m spacy download en

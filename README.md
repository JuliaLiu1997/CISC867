# Original Paper: Word class flexibility: A deep contextualized approach

This repository is the official implementation of "Word class flexibility: A deep contextualized approach" (https://arxiv.org/abs/2009.09241). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
git clone https://github.com/attardi/wikiextractor
pip install wikiextractor
```

## Datasets

1. Download ud treebanks:

```datasets
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz
mkdir data/ud_all
tar xf ud-treebanks-v2.5.tgz --directory data/ud_all
```

2. Download bnc baby:
```datasets
mkdir data/bnc
cd data/bnc
wget https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2553/2553.zip
!unzip 2553.zip

```

## Download and extract English (and other languages) Wikipedia

To Download and extract English Wikipedia, run:

```wiki
mkdir ../wiki
cd ./wiki
wget https://dumps.wikimedia.org/enwiki/20210220/enwiki-20210220-pages-articles-multistream1.xml-p1p41242.bz2
bzip2 -d enwiki*
python -m wikiextractor.WikiExtractor enwiki* -o en
cd ../..
```
Wikipedia for other language can also be found in https://dumps.wikimedia.org/


## Preprocess Wikipedia and BNC baby

To process English Wikipedia, run:

```wiki
python reproduction/process_wikipedia.py \
    --wiki_dir=data/wiki/ \
    --ud_dir=data/ud_all/ud-treebanks-v2.5/ \
    --dest_dir=data/wiki/\
    --lang=en \
    --model=stanza \
    --tokens 10000000
```

To process BNC baby, run:

```bnc
python reproduction/process_bnc.py \
    --bnc_dir=data/bnc/download/Texts \
    --to=data/bnc/bnc.pkl

```

## Run semantic metrics

To run semantic metrics:

```metrics
mkdir results
python reproduction/model_contextual.py \
      --pkl_dir data/wiki/processed_udpipe \
      --pkl_file en.pkl \
      --results_dir results/en/ \
      --model bert-base-multilingual-cased
```

>📋  model name options: "bert-base-multilingual-cased", "bert-base-uncased", "xlm-roberta-base", "elmo"

## Results

2 main results are :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)



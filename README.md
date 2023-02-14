# MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer

## __Repository unfinished__

##### Authors' code for paper "MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer", ACMMM 2022.

## Prerequisites

- Python == 3.9
- PyTorch == 1.12.1
- stanfordcorenlp == 4.2.2
- transformers == 4.12.5
- pycocoevalcap （https://github.com/sks3i/pycocoevalcap）

## Datasets
__VIST-E__: download the *SIS-with-labels.tar.gz* (https://visionandlanguage.net/VIST/dataset.html),  download the image features (https://vist-arel.s3.amazonaws.com/resnet_features.zip) and put then in `data/VIST-E`. 

__LSMDC-E__: download LSMDC 2021 version （*LSMDC16_annos_training_someone.csv*, *LSMDC16_annos_val_someone.csv*, *LSMDC16_annos_test_someone.csv*, *resnet152_200.zip*） (https://sites.google.com/site/describingmovies/home) and put then in `data/LSMDC-E`. __NOTE__: Due to LSMDC agreement, we cannot share data to any third-party.

We utilize Glove embedding, please download the *glove.6b.300d.txt* and put it in `data/`.

## Data preprocess

__VIST-E__:
1. Unzip *SIS-with-labels.tar.gz* to `data/VIST-E` and unzip *resnet_features.zip* to `data/VIST-E/image_features`.
2. Run `data/annotations.py`.
3. Run `data/img_feat_path.py `.
4. Run `data/pro_label.py`.
5. Run `data/embed_vocab.py` and make sure the parameters are set to *VIST-E*.

__LSMDC-E__:
1. Unzip all resnet features in *resnet152_200.zip* to a folder `data/LSMDC-E/image_features` without any subfolders.
2. Run `data/LSMDC-E/prepro_vocab.py`.
3. Run `data/embed_vocab.py` and make sure the parameters are set to *LSMDC-E*.


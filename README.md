# MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer

##### Authors' code for paper "MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer", ACMMM 2022.

## Prerequisites

- Python == 3.9
- PyTorch == 1.12.1
- stanfordcorenlp == 4.2.2
- transformers == 4.12.5
- pycocoevalcap （https://github.com/sks3i/pycocoevalcap）

## Quick start

- Dataset

All dataset under the directory of `data`. 

For __VIST-E__ dataset, download the SIS-with-labels.tar.gz (https://visionandlanguage.net/VIST/dataset.html),  download the image features (https://vist-arel.s3.amazonaws.com/resnet_features.zip) and put then in directory of `data/VIST-E`. 

For __LSMDC-E__ dataset, download LSMDC 2021 version （LSMDC16_annos_training_someone.csv, LSMDC16_annos_val_someone.csv, LSMDC16_annos_test_someone.csv） (https://sites.google.com/site/describingmovies/home) and put then in directory of `data/LSMDC-E`. __NOTE__: Due to LSMDC agreement, we cannot share data to any third-party.

We utilize the glove embedding, please download the *glove.6b.300d.txt* and put it in directory of `data/`.
- Data preprocess

Run following command:

# MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer

##### Authors' code for paper "MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer", ACMMM 2022.

## Prerequisites

- Python 3.9
- PyTorch == 1.12.1
- stanfordcorenlp 4.2.2
- transformers 4.12.5

## Quick start:

- Dataset

All dataset under the directory of `data`. 

For __VIST-E__ dataset, download the SIS-with-labels.tar.gz (https://visionandlanguage.net/VIST/dataset.html),  download the image features (https://vist-arel.s3.amazonaws.com/resnet_features.zip) and put then in directory of `data`. 

For __LSMDC-E__ dataset, download LSMDC 2021 version (https://sites.google.com/site/describingmovies/home)

We utilize the glove embedding, please download the *glove.6b.300d.txt* and put it in directory of `data/{dataset}/embedding`.
- Data preprocess

Run following command:

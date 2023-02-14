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
1. Unzip *SIS-with-labels.tar.gz* to `data/VIST-E`.
2. Unzip conv features in *resnet_features.zip* to a folder `data/VIST-E/image_features` without any subfolders..
3. Run `data/annotations.py`.
4. Run `data/img_feat_path.py `.
5. Run `data/pro_label.py`.
6. Run `data/embed_vocab.py` and make sure the parameters are set to *VIST-E*.

__LSMDC-E__:
1. Unzip all resnet features in *resnet152_200.zip* to a folder `data/LSMDC-E/image_features` without any subfolders.
2. Run `data/LSMDC-E/prepro_vocab.py`.
3. Run `data/embed_vocab.py` and make sure the parameters are set to *LSMDC-E*.

## Process
1. Set parameters in `utils/opts.py`.
2. Run `train.py` to train a model.
3. Run `eval.py` to evaluate a model.

## Citation
If you find our work or the code useful, please consider cite our paper using:
```bash
@inproceedings{10.1145/3503161.3548022,
author = {Xue, Dizhan and Qian, Shengsheng and Fang, Quan and Xu, Changsheng},
title = {MMT: Image-Guided Story Ending Generation with Multimodal Memory Transformer},
year = {2022},
doi = {10.1145/3503161.3548022},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {750–758},
numpages = {9},
}
```

# MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer

##### Authors' code for paper "MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer", ACMMM 2022.

![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mmt-image-guided-story-ending-generation-with/image-guided-story-ending-generation-on-vist) ![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mmt-image-guided-story-ending-generation-with/image-guided-story-ending-generation-on-lsmdc)

## Prerequisites

- Python == 3.9
- PyTorch == 1.12.1
- stanfordcorenlp == 3.9.1.1 with *stanford-corenlp-4.2.2*
- transformers == 4.12.5
- pycocoevalcap （https://github.com/sks3i/pycocoevalcap）

## Datasets
__VIST-E__: download the *SIS-with-labels.tar.gz* (https://visionandlanguage.net/VIST/dataset.html),  download the image features (https://vist-arel.s3.amazonaws.com/resnet_features.zip) and put them in `data/VIST-E`. 

__LSMDC-E__: download LSMDC 2021 version （*task1_2021.zip*, *resnet152_200.zip*） (https://sites.google.com/site/describingmovies/home) and put them in `data/LSMDC-E`. __NOTE__: Due to LSMDC agreement, we cannot share data to any third-party.

We utilize Glove embedding, please download the *glove.6b.300d.txt* and put it in `data/`.

## Data Preprocess

__VIST-E__:
1. Unzip *SIS-with-labels.tar.gz* to `data/VIST-E`.
2. Unzip conv features in *resnet_features.zip* to a folder `data/VIST-E/image_features` without any subfolders.
3. Run `data/VIST-E/annotations.py`.
4. Run `data/VIST-E/img_feat_path.py `.
5. Run `data/VIST-E/pro_label.py`.
6. Run `data/embed_vocab.py` and make sure parameter `dataset` is set to *VIST-E*.

__LSMDC-E__:
1. Unzip *task1_2021.zip* to `data/LSMDC-E`.
2. Unzip all resnet features in *resnet152_200.zip* to a folder `data/LSMDC-E/image_features` without any subfolders.
3. Run `data/LSMDC-E/prepro_vocab.py`.
4. Run `data/embed_vocab.py` and make sure parameter `dataset` is set to *LSMDC-E*.

## Process
1. Set parameters in `utils/opts.py`.
2. Run `train.py` to train a model.
3. Run `eval.py` to evaluate a model.

    __Recommended Settings__

    VIST-E w BERT:
    ```bash
    python train.py --dataset VIST-E --use_bert True --num_head 4 --weight_decay 0 --grad_clip_value 0
    ```
    VIST-E w/o BERT:
    ```bash
    python train.py --dataset VIST-E --use_bert False --num_head 4 --weight_decay 1e-5 --grad_clip_value 0
    ```
    LSMDC-E w BERT:
    ```bash
    python train.py --dataset LSMDC-E --use_bert True --num_head 8 --weight_decay 1e-5 --grad_clip_value 0.1
    ```
    LSMDC-E w/o BERT:
    ```bash
    python train.py --dataset LSMDC-E --use_bert False --num_head 8 --weight_decay 1e-5 --grad_clip_value 0.1
    ```

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

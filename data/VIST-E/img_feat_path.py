import os
from glob import glob
import json

images_directory_train = 'resnet_features/conv/train'
images_directory_val = 'resnet_features/conv/val'
images_directory_test = 'resnet_features/conv/test'
features_num = 'feat_num.json'

images_path_names_train = [y for x in os.walk(images_directory_train) for y in glob(os.path.join(x[0], "*.npz"))]
images_path_names_val = [y for x in os.walk(images_directory_val) for y in glob(os.path.join(x[0], "*.npz"))]
images_path_names_test = [y for x in os.walk(images_directory_test) for y in glob(os.path.join(x[0], "*.npz"))]

feat_num = []
print(len(images_path_names_train))         # 209639 # 167524
print(len(images_path_names_val))           #  21043
print(len(images_path_names_test))          #  21072
for i, path in enumerate(images_path_names_train):
    feat_s = path.split('/')
    feat_num_npz = feat_s[-1]
    feat_num_s = feat_num_npz.split('.')
    feat_num_s = feat_num_s[0]
    feat_num.append(feat_num_s)

for i, path in enumerate(images_path_names_val):
    feat_s = path.split('/')
    feat_num_npz = feat_s[-1]
    feat_num_s = feat_num_npz.split('.')
    feat_num_s = feat_num_s[0]
    feat_num.append(feat_num_s)

for i, path in enumerate(images_path_names_test):
    feat_s = path.split('/')
    feat_num_npz = feat_s[-1]
    feat_num_s = feat_num_npz.split('.')
    feat_num_s = feat_num_s[0]
    feat_num.append(feat_num_s)

json.dump(feat_num, open(features_num, 'w'))

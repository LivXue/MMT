import os
from glob import glob
import json

images_directory = 'image_features'
features_num = 'feat_num.json'

images_path_names = [y for x in os.walk(images_directory) for y in glob(os.path.join(x[0], "*.npz"))]

feat_num = []
print(len(images_path_names))         # 209639
for i, path in enumerate(images_path_names):
    feat_s = path.split('/')
    feat_num_npz = feat_s[-1]
    feat_num_s = feat_num_npz.split('.')
    feat_num_s = feat_num_s[0]
    feat_num.append(feat_num_s)

json.dump(feat_num, open(features_num, 'w'))

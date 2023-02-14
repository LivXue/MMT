from tqdm import tqdm
import numpy as np

import json
import operator


Data = ['train', 'val', 'test']
storys = []
for data in Data:
    # storys = []

    f = json.load(open('%s.story-in-sequence.json' % data))
    # print(type(f))
    annotations = f['annotations']
    print(type(annotations))
    print(len(annotations))
    for annotation in annotations:
        anno = annotation[0]

        # print(type(anno))
        if data == 'train':
            anno['split'] = 'train'
            storys.append([anno])
            # print(anno)
        elif data == 'val':
            anno['split'] = 'val'
            storys.append([anno])
            # print(anno)
        else:
            anno['split'] = 'test'
            storys.append([anno])
            # print(anno)
print(storys)

json.dump(storys, open('annotation.json', 'w'))
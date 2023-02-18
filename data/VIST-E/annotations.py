import json


Data = ['train', 'val', 'test']
storys = []
for data in Data:
    f = json.load(open('%s.story-in-sequence.json' % data))
    annotations = f['annotations']
    print(len(annotations))     # 200775 # 24950 # 25275
    for annotation in annotations:
        anno = annotation[0]
        if data == 'train':
            anno['split'] = 'train'
            storys.append([anno])
        elif data == 'val':
            anno['split'] = 'val'
            storys.append([anno])
        else:
            anno['split'] = 'test'
            storys.append([anno])

json.dump(storys, open('annotation.json', 'w'))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import json
import numpy as np
import operator
import h5py
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP

path = '../stanford-corenlp-4.2.2'
nlp = StanfordCoreNLP(path)
print("Successfully loaded Stanford-CoreNLP models!")


def story_pro(annotations, params):
    print('len_annotations: ', len(annotations))

    image_feat_num = params['image_features']
    img_feat_num = json.load(open(image_feat_num, 'r'))
    print('len_feat_num: ', len(img_feat_num))
    count = 0
    story = []

    for i in tqdm(range(0, len(annotations), 5)):
        try:
            story_id = annotations[i][0]['story_id']

            img_id1, order1 = int(annotations[i][0]["photo_flickr_id"]), annotations[i][0][
                "worker_arranged_photo_order"]
            img_id2, order2 = int(annotations[i + 1][0]["photo_flickr_id"]), annotations[i + 1][0][
                "worker_arranged_photo_order"]
            img_id3, order3 = int(annotations[i + 2][0]["photo_flickr_id"]), annotations[i + 2][0][
                "worker_arranged_photo_order"]
            img_id4, order4 = int(annotations[i + 3][0]["photo_flickr_id"]), annotations[i + 3][0][
                "worker_arranged_photo_order"]
            img_id5, order5 = int(annotations[i + 4][0]["photo_flickr_id"]), annotations[i + 4][0][
                "worker_arranged_photo_order"]

            if not (str(img_id5) in img_feat_num):
                count += 1
                print('img_id5', img_id5)
                print(count)
                continue
            else:
                img_str_id5 = str(img_id5)

            story1 = annotations[i][0]['text']
            story2 = annotations[i + 1][0]['text']
            story3 = annotations[i + 2][0]['text']
            story4 = annotations[i + 3][0]['text']
            story5 = annotations[i + 4][0]['text']

            story1_token = nlp.word_tokenize(story1)
            story2_token = nlp.word_tokenize(story2)
            story3_token = nlp.word_tokenize(story3)
            story4_token = nlp.word_tokenize(story4)
            story5_token = nlp.word_tokenize(story5)

            if len(story1_token) > params['max_length']:
                count += 1
                continue
            if len(story2_token) > params['max_length']:
                count += 1
                continue
            if len(story3_token) > params['max_length']:
                count += 1
                continue
            if len(story4_token) > params['max_length']:
                count += 1
                continue
            if len(story5_token) > params['max_length']:
                count += 1
                continue

            story_list = [(story1, order1), (story2, order2), (story3, order3), (story4, order4), (story5, order5)]
            story_list = sorted(story_list, key=operator.itemgetter(1))

            story_token_list = [(story1_token, order1), (story2_token, order2), (story3_token, order3),
                                (story4_token, order4), (story5_token, order5)]
            story_token_list = sorted(story_token_list, key=operator.itemgetter(1))

            story_token_list = [story_token_list[0][0], story_token_list[1][0], story_token_list[2][0],
                                story_token_list[3][0], story_token_list[4][0]]

            img_id_list = [(img_id1, order1), (img_id2, order2), (img_id3, order3), (img_id4, order4),
                           (img_str_id5, order5)]
            img_id_list = sorted(img_id_list, key=operator.itemgetter(1))

            ordered_stories_four = [story_list[0][0], story_list[1][0], story_list[2][0], story_list[3][0]]
            ordered_stories_last = story_list[4][0]
            order_last_img_id = img_id_list[4][0]
            story.append({'story_id': story_id,
                          'story_token_list': story_token_list,
                          'stories_four': ordered_stories_four,#story_rep_four,
                          'stories_last': ordered_stories_last,
                          'last_img_id': order_last_img_id,
                          'split': annotations[i][0]['split']})
        except json.decoder.JSONDecodeError:
            continue
    nlp.close()
    with open('story.json', 'w') as ff:
        json.dump(story, ff)

    return story


def build_vocab(storys, params):
    count_thr = params['word_count_threshold_test']

    counts = {}
    for story in storys:
        last_token = story['story_token_list'][4]
        for w in last_token:
            counts[w] = counts.get(w, 0) + 1
    vocab = ['[SOS]']
    vocab.extend([w for w, n in counts.items() if n > count_thr])

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for story in storys:
        token = story['story_token_list']
        txt = token[-1]
        nw = len(txt)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1

    # lets now produce the final annotations
    vocab.append('[UNK]')
    vocab.append('[male]')
    vocab.append('[female]')

    for story in storys:
        sents = story['stories_last']
        last = []
        for i, word in enumerate(sents.split(' ')):
            if word in vocab:
                last.append(word)
            else:
                last.append('[UNK]')
        story['story_end'] = last

    return vocab


def build_src_vocab(storys, params):
    count_thr = params['word_count_threshold']

    counts = {}
    for story in storys:
        token = story['story_token_list'][:4]
        for sentence in token:
            for w in sentence:
                counts[w] = counts.get(w, 0) + 1

    vocab = ['[UUD]']
    vocab.extend([w for w, n in counts.items() if n > count_thr])

    # lets now produce the final annotations
    vocab.append('[UNK]')
    vocab.append('[male]')
    vocab.append('[female]')
    return vocab


def encode_story_four(storys, params, src_wtoi):
    max_length = 40
    first = []
    second = []
    third = []
    four = []

    for i, story in enumerate(storys):
        sent_insts = story['stories_four']
        for j, sents in enumerate(sent_insts):
            if j == 0:
                sent_lsit = np.zeros((40), dtype='uint32')
                for i, word in enumerate(sents.split(' ')):
                    if i < max_length:
                        if word in src_wtoi:
                            sent_lsit[i] = (src_wtoi[word])
                        else:
                            sent_lsit[i] = (src_wtoi['[UNK]'])
                    else:
                        break
                first.append(sent_lsit)
            elif j == 1:
                sent_lsit = np.zeros((40), dtype='uint32')
                for i, word in enumerate(sents.split(' ')):
                    if i < max_length:
                        if word in src_wtoi:
                            sent_lsit[i] = (src_wtoi[word])
                        else:
                            sent_lsit[i] = (src_wtoi['[UNK]'])
                    else:
                        break
                second.append(sent_lsit)
            elif j == 2:
                sent_lsit = np.zeros((40), dtype='uint32')
                for i, word in enumerate(sents.split(' ')):
                    if i < max_length:
                        if word in src_wtoi:
                            sent_lsit[i] = (src_wtoi[word])
                        else:
                            sent_lsit[i] = (src_wtoi['[UNK]'])
                    else:
                        break
                third.append(sent_lsit)
            else:
                sent_lsit = np.zeros((40), dtype='uint32')
                for i, word in enumerate(sents.split(' ')):
                    if i < max_length:
                        if word in src_wtoi:
                            sent_lsit[i] = (src_wtoi[word])
                        else:
                            sent_lsit[i] = (src_wtoi['[UNK]'])
                    else:
                        break
                four.append(sent_lsit)

    return first, second, third, four


def encode_story_last(storys, params, wtoi):
    max_length = 40
    N = len(storys)
    M = len(storys)

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    story_counter = 0
    counter = 1

    for i, story in enumerate(storys):
        Li = np.zeros((1, max_length), dtype='uint32')
        s = story['story_end']
        label_length[story_counter] = min(max_length, len(s))
        story_counter += 1
        for k, w in enumerate(s):
            if k < max_length:
                Li[0, k] = wtoi[w]

        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter

        counter += 1

    L = np.concatenate(label_arrays, axis=0)
    assert L.shape[0] == M, 'length don\'t match that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    return L, label_start_ix, label_end_ix, label_length


def main(params):
    file = open(params['input_json'], 'r')
    data = json.load(file)
    anno = data
    print(len(data))
    print(anno[1])
    #story = story_pro(anno, params)
    story = json.load(open("story.json"))
    file.close()
    # file = open('story5.json', 'r')
    # story = json.load(file)
    # file.close()

    vocab = build_vocab(story, params)
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}

    src_vocab = build_src_vocab(story, params)
    src_itow = {i: w for i, w in enumerate(src_vocab)}
    src_wtoi = {w: i for i, w in enumerate(src_vocab)}

    first, second, third, four = encode_story_four(story, params, src_wtoi)
    # print('first_len: ', len(first))
    f_fe = h5py.File(params['output_h5_fe'] + '_label.h5', 'w')
    f_fe.create_dataset('sent1', dtype='uint32', data=first)
    f_fe.create_dataset('sent2', dtype='uint32', data=second)
    f_fe.create_dataset('sent3', dtype='uint32', data=third)
    f_fe.create_dataset('sent4', dtype='uint32', data=four)
    f_fe.close()
    print('Saved story_four_wtoi!')

    L, label_start_ix, label_end_ix, label_length = encode_story_last(story, params, wtoi)
    print('L_len:', len(L))
    f_lb = h5py.File(params['output_h5'] + '_label.h5', 'w')
    f_lb.create_dataset('labels', dtype='uint32', data=L)
    f_lb.create_dataset('label_start_ix', dtype='uint32', data=label_start_ix)
    f_lb.create_dataset('label_end_ix', dtype='uint32', data=label_end_ix)
    f_lb.create_dataset('label_length', dtype='uint32', data=label_length)
    f_lb.close()
    print('Down f_lb')

    story_four = []
    for i, imgs in enumerate(story):
        story_four.append(imgs['stories_four'])
    with open(params['output_story_four'], 'w') as f_four:
        json.dump(story_four, f_four)
    f_four.close()
    print('Saved story_four!')

    out = {}
    out['src_word_to_ix'] = src_wtoi
    out['src_ix_to_word'] = src_itow
    out['tgt_word_to_ix'] = wtoi
    out['tgt_ix_to_word'] = itow

    out['images'] = []
    for i, img in enumerate(story):
        jimg = {}
        jimg['split'] = img['split']
        jimg['id'] = img['last_img_id']
        jimg['story_id'] = img['story_id']
        jimg['story_end'] = img['stories_last']
        out['images'].append(jimg)

    with open(params['output_json'], 'w') as ff:
        json.dump(out, ff)
    ff.close()

    # json.dump(out, open(params['output_json'], 'w'))
    print('wrote', params['output_json'])
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='annotation.json', help='')
    parser.add_argument('--output_json', default='data_res.json', help='')
    parser.add_argument('--output_h5', default='data_tgt', help='')
    parser.add_argument('--output_story_four', default='data_four.json', help='')
    parser.add_argument('--output_h5_fe', default='data_src', help='')
    parser.add_argument('--image_features', default='feat_num.json', help='')

    parser.add_argument('--max_length', default=40, type=int, help='')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='')
    parser.add_argument('--word_count_threshold_test', default=1, type=int, help='')

    args = parser.parse_args()
    params = vars(args)

    main(params)

# nlp.close()

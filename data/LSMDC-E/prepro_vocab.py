import re
import json
import argparse
import os
import csv

import numpy as np
import h5py
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm


path = 'stanford-corenlp-4.2.2'
nlp = StanfordCoreNLP(path)
print("Successfully loaded {} models!".format(path))


def collect_story(params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    max_len = []
    csvs = ['LSMDC16_annos_training_someone.csv', 'LSMDC16_annos_val_someone.csv', 'LSMDC16_annos_test_someone.csv']
    for c in csvs[:-1]:
        with open(os.path.join(params['input_path'], c)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                # remove punctuation but keep possessive because we want to separate out character names
                ws = re.sub(r'[.!,;?]', ' ', str(row[5]).lower()).replace("'s", " 's").split()
                max_len.append(len(ws))
                for w in ws:
                    counts[w] = counts.get(w, 0) + 1
    print('max count', np.mean(max_len))
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('[UNK]')

    splits = ['train', 'val', 'test']
    videos = []
    movie_ids = {}
    vid = 0
    groups = []
    gid = -1
    for i, c in enumerate(csvs):
        split = splits[i]
        with open(os.path.join(params['input_path'], c)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                clip = row[0]
                movie = clip[:clip.rfind('_')]
                info = {'id': vid, 'split': split, 'movie': movie, 'clip': clip}
                if movie not in movie_ids:
                    gid += 1
                    ginfo = {'id': gid, 'split': split, 'movie': movie, 'videos': [vid]}
                    groups.append(ginfo)
                    gcount = 0
                    movie_ids[movie] = [gid]
                else:
                    if gcount >= params['group_by']:
                        gid += 1
                        ginfo = {'id': gid, 'split': split, 'movie': movie, 'videos': [vid]}
                        groups.append(ginfo)
                        gcount = 0
                        movie_ids[movie].append(gid)
                    else:
                        groups[gid]['videos'].append(vid)
                if split != 'blind_test':
                    ws = re.sub(r'[.!,;?]', ' ', str(row[5]).lower()).replace("'s", " 's").split()
                    caption = [w if counts.get(w, 0) > count_thr else '[UNK]' for w in ws]
                    if len(caption) > params['max_length']:
                        caption = caption[:params['max_length']]
                    info['final_caption'] = ' '.join(caption)
                videos.append(info)
                vid += 1
                gcount += 1
    return videos, groups, movie_ids, vocab


def build_vocab(storys, params):
    count_thr = params['word_count_threshold_test']

    counts = {}
    for story in storys:
        last_token = story['story_token_list'][4]
        for w in last_token:
            counts[w] = counts.get(w, 0) + 1
    vocab = ['[SOS]']
    vocab.extend([w for w, n in counts.items() if n > count_thr])

    # lets now produce the final annotations
    vocab.append('[UNK]')

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


def story_pro(videos, groups, params):
    story = []
    for ginfo in tqdm(groups):
        if len(ginfo['videos']) != 5:
            continue
        story1 = videos[ginfo['videos'][0]]['final_caption']
        story2 = videos[ginfo['videos'][1]]['final_caption']
        story3 = videos[ginfo['videos'][2]]['final_caption']
        story4 = videos[ginfo['videos'][3]]['final_caption']
        story5 = videos[ginfo['videos'][4]]['final_caption']
        stories_four = [story1, story2, story3, story4]

        last_seg_id = videos[ginfo['videos'][4]]['clip']
        split = videos[ginfo['videos'][4]]['split']

        story1_taken = nlp.word_tokenize(story1)
        story2_taken = nlp.word_tokenize(story2)
        story3_taken = nlp.word_tokenize(story3)
        story4_taken = nlp.word_tokenize(story4)
        story5_taken = nlp.word_tokenize(story5)
        story_token_list = [story1_taken, story2_taken, story3_taken, story4_taken, story5_taken]

        story1_depen = story1[:min(len(story1), params['max_length'])]
        story2_depen = story2[:min(len(story2), params['max_length'])]
        story3_depen = story3[:min(len(story3), params['max_length'])]
        story4_depen = story4[:min(len(story4), params['max_length'])]
        sent1 = nlp.dependency_parse(story1_depen)
        sent2 = nlp.dependency_parse(story2_depen)
        sent3 = nlp.dependency_parse(story3_depen)
        sent4 = nlp.dependency_parse(story4_depen)
        # sent5 = nlp.dependency_parse(story5)
        depen_list = [sent1, sent2, sent3, sent4]

        story.append({'story_id': ginfo['id'],
                      'story_token_list': story_token_list,
                      'stories_four': stories_four,  # story_rep_four,
                      'sent_depen': depen_list,
                      'stories_last': story5,
                      'last_img_id': last_seg_id,
                      'split': split})

    nlp.close()
    with open('story.json', 'w') as ff:
        json.dump(story, ff)

    return story


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
    return vocab


def encode_story_four(storys, params, src_wtoi):
    max_length = 20
    first = []
    second = []
    third = []
    four = []

    for i, story in enumerate(storys):
        sent_insts = story['stories_four']
        for j, sents in enumerate(sent_insts):
            if j == 0:
                sent_lsit = np.zeros((max_length), dtype='uint32')
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
                sent_lsit = np.zeros((max_length), dtype='uint32')
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
                sent_lsit = np.zeros((max_length), dtype='uint32')
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
                sent_lsit = np.zeros((max_length), dtype='uint32')
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
    max_length = 20
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
    # create the vocab
    #videos, groups, movie_ids, vocab = collect_story(params)
    #story = story_pro(videos, groups, params)
    story = json.load(open("story.json"))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_path', type=str, default='./',
                        help='directory containing csv files')
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--max_length', default=20, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--group_by', default=5, type=int,
                        help='group # of clips as one video')
    parser.add_argument('--output_json', default='data_res.json', help='')
    parser.add_argument('--output_h5', default='data_tgt', help='')
    parser.add_argument('--output_story_four', default='data_four.json', help='')
    parser.add_argument('--output_h5_fe', default='data_src', help='')
    parser.add_argument('--word_count_threshold_test', default=1, type=int, help='')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)

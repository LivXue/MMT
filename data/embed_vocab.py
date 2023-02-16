from __future__ import division
import argparse

import six
import numpy as np
import torch
import json


def get_vocabs(dict_path):
    fields = json.load(open(dict_path, 'r'))
    vocs = []
    for side in ['src_word_to_ix', 'tgt_word_to_ix']:
        try:
            vocab = fields[side]
        except AttributeError:
            vocab = fields[side]

        vocs.append(vocab)

    enc_vocab, dec_vocab = vocs[0], vocs[1]

    print("From: %s" % dict_path)
    print("\t* source vocab: %d words" % len(enc_vocab))    # LSMDC-E: 20699    # VIST-E: 26641
    print("\t* target vocab: %d words" % len(dec_vocab))    # LSMDC-E: 7275     # VIST-E: 8745

    return enc_vocab, dec_vocab


def read_embeddings(file_enc, skip_lines=0):
    """
    :param file_enc:
    :param skip_lines:
    :return:
    """
    embs = dict()
    with open(file_enc, 'rb') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                continue

            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    return embs


def match_embeddings(vocab, emb, opt):
    """
    :param vocab:
    :param emb:
    :param opt:
    :return:
    """
    dim = len(six.next(six.itervalues(emb)))
    print('embedding_dim:', dim)
    print('vocab: ', len(vocab))
    filtered_embeddings = np.zeros((len(vocab), dim))
    print('filtered_embeddings: ', filtered_embeddings.shape)
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.items():
        if w in emb:
            # print('w_id: ', w_id)
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        elif w == '[male]':
            print("Found [male]!")
            filtered_embeddings[w_id] = [x + y for x, y in zip(emb['man'], emb['name'])]
        elif w == '[female]':
            print("Found [female]!")
            filtered_embeddings[w_id] = [x + y for x, y in zip(emb['woman'], emb['name'])]
        else:
            print("Missing word: {}".format(w))
            filtered_embeddings[w_id] = np.random.uniform(-0.1, 0.1, dim).astype('f')
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


def main():
    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('--dataset', type=str, default='LSMDC-E',
                        help='dataset: VIST-E / LSMDC-E')
    parser.add_argument('-emb_file_enc', type=str, default='glove.6B.300d.txt',
                        help="source Embeddings from this file")
    parser.add_argument('-emb_file_dec', type=str, default='glove.6B.300d.txt',
                        help="target Embeddings from this file")
    parser.add_argument('-verbose', action="store_true", default=False)
    parser.add_argument('-skip_lines', type=int, default=0,
                        help="Skip first lines of the embedding file")
    parser.add_argument('-type', choices=["GloVe", "word2vec"],
                        default="GloVe")
    opt = parser.parse_args()

    enc_vocab, dec_vocab = get_vocabs(opt.dataset + '/data_res.json')

    skip_lines = 1 if opt.type == "word2vec" else opt.skip_lines
    src_vectors = read_embeddings(opt.emb_file_enc, skip_lines)

    tgt_vectors = read_embeddings(opt.emb_file_dec)

    filtered_enc_embeddings, enc_count = match_embeddings(
        enc_vocab, src_vectors, opt)
    filtered_dec_embeddings, dec_count = match_embeddings(
        dec_vocab, tgt_vectors, opt)

    enc_output_file = opt.dataset + "/embedding/embedding_enc.pt"
    dec_output_file = opt.dataset + "/embedding/embedding_dec.pt"

    torch.save(filtered_enc_embeddings, enc_output_file)
    torch.save(filtered_dec_embeddings, dec_output_file)


if __name__ == "__main__":
    main()

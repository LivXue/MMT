from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import torch
import torch.utils.data as data
import numpy as np
import numpy.random as npr
import json
import h5py
import lmdb
import six

from dataloader.prefetch_dataloader import DataLoaderX


class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """

    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                # Normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.
                return x['arr_0'] if 'arr_0' in x else x['z']

            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                                 readonly=True, lock=False,
                                 readahead=False, meminit=False)
        elif db_path.endswith('.pth'):  # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}

    def get(self, key):
        # print('key: ', key)

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(key.encode())
            f_input = byteflow
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # Load image
        feat = self.loader(f_input)

        return feat


class Dataset(data.Dataset):

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_src_vocab_size(self):
        return self.src_vocab_size

    def get_src_vocab(self):
        return self.src_ix_to_word

    def __init__(self, opt, info):
        self.opt = opt
        # self.seq_per_img = opt.seq_per_img
        self.seq_per_img = 1  # Number of captions for one image
        self.split_ix = None

        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_conv = getattr(opt, 'use_conv', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_conv_feat = getattr(opt, 'norm_conv_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        self.info = info
        if 'tgt_ix_to_word' in self.info:
            self.ix_to_word = self.info['tgt_ix_to_word']
            self.vocab_size = len(self.ix_to_word)

        self.src_ix_to_word = self.info['src_ix_to_word']
        self.src_vocab_size = len(self.src_ix_to_word)
        self.src_word_to_ix = self.info['src_word_to_ix']

        # Open the hdf5 file.
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(opt.input_label_h5, 'r', driver='core')
            # Load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            # Load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        else:
            self.seq_length = 1

        # Load src
        if opt.use_bert and os.path.exists(opt.input_src_h5 + '.bert'):
            self.src = h5py.File(opt.input_src_h5 + '.bert', 'r', driver='core')
        else:
            self.src = h5py.File(opt.input_src_h5, 'r', driver='core')
        self.sent1 = self.src['sent1'][:]
        self.sent2 = self.src['sent2'][:]
        self.sent3 = self.src['sent3'][:]
        self.sent4 = self.src['sent4'][:]
        # if not exist, save BERT src (word index)
        if opt.use_bert and not os.path.exists(opt.input_src_h5 + '.bert'):
            self.convert_to_bert_ids(opt)
            f_fe = h5py.File(opt.input_src_h5 + '.bert', 'w')
            f_fe.create_dataset('sent1', dtype='uint32', data=self.sent1)
            f_fe.create_dataset('sent2', dtype='uint32', data=self.sent2)
            f_fe.create_dataset('sent3', dtype='uint32', data=self.sent3)
            f_fe.create_dataset('sent4', dtype='uint32', data=self.sent4)
            f_fe.close()

        # Initialize image feature loaders
        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        self.fc_loader = HybridLoader(opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        self.conv_loader = HybridLoader(opt.input_conv_dir, opt.conv_ext, in_memory=self.data_in_memory)
        self.box_loader = HybridLoader(opt.input_box_dir, '.npy', in_memory=self.data_in_memory)

    def convert_to_bert_ids(self, opt):
        self.src_ix_to_word[str(self.src_word_to_ix['[UUD]'])] = '[PAD]'
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained(opt.bert_architecture)
        seq_len = self.sent1.shape[1]
        sent1 = np.zeros((self.sent1.shape[0], seq_len + 2), dtype=self.sent1.dtype)
        sent2 = np.zeros((self.sent2.shape[0], seq_len + 2), dtype=self.sent2.dtype)
        sent3 = np.zeros((self.sent3.shape[0], seq_len + 2), dtype=self.sent3.dtype)
        sent4 = np.zeros((self.sent4.shape[0], seq_len + 2), dtype=self.sent4.dtype)

        def sent_to_bert_ids(input_sent):
            sent = [self.src_ix_to_word[str(x)] for x in input_sent]
            sent = " ".join(sent)
            sent = bert_tokenizer.tokenize(sent)
            sent = bert_tokenizer.convert_tokens_to_ids(sent)[:seq_len]
            try:
                tail = sent.index(0)
                sent.insert(tail, bert_tokenizer.convert_tokens_to_ids('[SEP]'))
            except ValueError:
                sent.append(bert_tokenizer.convert_tokens_to_ids('[SEP]'))

            sent.insert(0, bert_tokenizer.convert_tokens_to_ids('[CLS]'))
            sent = np.array(sent).astype(input_sent.dtype)
            return sent

        for i in range(self.sent1.shape[0]):
            sent1[i] = sent_to_bert_ids(self.sent1[i])
        for i in range(self.sent2.shape[0]):
            sent2[i] = sent_to_bert_ids(self.sent2[i])
        for i in range(self.sent3.shape[0]):
            sent3[i] = sent_to_bert_ids(self.sent3[i])
        for i in range(self.sent4.shape[0]):
            sent4[i] = sent_to_bert_ids(self.sent4[i])

        self.sent1 = sent1
        self.sent2 = sent2
        self.sent3 = sent3
        self.sent4 = sent4

    def get_captions(self, ix, seq_per_img):
        # Fetch the sequence labels.
        ix1 = self.label_start_ix[ix] - 1  # Label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # Number of captions available for this image
        assert ncap > 0, 'An image does not have any label. tThis can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # We need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = npr.randint(ix1, ix2 + 1)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = npr.randint(ix1, ix2 - seq_per_img + 2)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_src(self, ix):
        return self.sent1[ix], self.sent2[ix], self.sent3[ix], self.sent4[ix]

    def collate_func(self, batch):
        seq_per_img = self.seq_per_img

        fc_batch, conv_batch, label_batch = [], [], []
        src1_batch, src2_batch, src3_batch, src4_batch = [], [], [], []
        infos = []
        gts = []

        for sample in batch:
            # Fetch image
            tmp_fc, tmp_conv, tmp_seq, ix, tem_src1, tem_src2, tem_src3, tem_src4 = sample

            fc_batch.append(tmp_fc)
            conv_batch.append(tmp_conv)

            src1_batch.append(tem_src1)
            src2_batch.append(tem_src2)
            src3_batch.append(tem_src3)
            src4_batch.append(tem_src4)

            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype='int')
            if hasattr(self, 'h5_label_file'):
                # If there is ground truth
                tmp_label[:, 1: self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # If there is ground truth
                gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])

            # Record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)

        data = {'src1': np.stack(src1_batch).astype(int), 'src2': np.stack(src2_batch).astype(int),
                'src3': np.stack(src3_batch).astype(int), 'src4': np.stack(src4_batch).astype(int),
                'fc_feats': np.stack(fc_batch)}

        # Merge conv_feats
        max_conv_len = max([_.shape[0] for _ in conv_batch])
        # print('len(conv_batch): ', len(conv_batch))
        data['conv_feats'] = np.zeros([len(conv_batch), max_conv_len, conv_batch[0].shape[1]], dtype='float32')
        data['conv_masks'] = np.zeros(data['conv_feats'].shape[:2], dtype=np.bool)  # size(batch, numbers_conv)
        for i in range(len(conv_batch)):
            data['conv_feats'][i, :conv_batch[i].shape[0]] = conv_batch[i]
            data['conv_masks'][i, :conv_batch[i].shape[0]] = 1

        # Set conv_masks to None if convolutional features have same length
        if data['conv_masks'].sum() == data['conv_masks'].size:
            data['conv_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # Generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts  # all ground truth captions of each images
        data['infos'] = infos

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in data.items()}
        return data

    def __getitem__(self, index):
        """
        This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        if self.use_conv:
            conv_feat = self.conv_loader.get(str(self.info['images'][ix]['id']))
            # Reshape to K x C
            conv_feat = conv_feat.reshape(-1, conv_feat.shape[-1])
            if self.norm_conv_feat:
                conv_feat = conv_feat / np.linalg.norm(conv_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
                # Devided by image width and height
                x1, y1, x2, y2 = np.hsplit(box_feat, 4)
                h, w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack(
                    (x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))  # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                conv_feat = np.hstack([conv_feat, box_feat])
                # Sort the features by the size of boxes
                conv_feat = np.stack(sorted(conv_feat, key=lambda x: x[-1], reverse=True))
        else:
            conv_feat = np.zeros((0, 0), dtype='float32')
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of convltional features when there is no fc provided (For bottomup feature)
                fc_feat = conv_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None

        # src
        src1, src2, src3, src4 = self.get_src(ix)

        return (fc_feat, conv_feat, seq, ix, src1, src2, src3, src4)

    def __len__(self):
        return len(self.split_ix)


class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size

        # Load the json file which contains additional information about the dataset
        self.info = json.load(open(self.opt.input_json))

        # Initialize datasets
        self.datasets = {x: Dataset(opt, self.info) for x in ['train', 'val', 'test']}

        self.num_images = len(self.info['images'])  # self.label_start_ix.shape[0]
        print('Read %d image features' % self.num_images)

        # Separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if 'split' not in img.keys():
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            else:
                self.split_ix[img['split']].append(ix)

        print('Assigned %d samples to split train.' % len(self.split_ix['train']))
        print('Assigned %d samples to split val.' % len(self.split_ix['val']))
        print('Assigned %d samples to split test.' % len(self.split_ix['test']))

        # Initialize loaders
        self.loaders = {}
        for split in ['train', 'val', 'test']:
            self.datasets[split].split_ix = self.split_ix[split]
            if split == 'train':
                sampler = MySampler(self.datasets[split].split_ix, shuffle=True)
            else:
                sampler = MySampler(self.datasets[split].split_ix, shuffle=False)

            # sampler is an iter generates keys for fetching the batch. Generate one key every time
            # collate_fn processes the batch outputted by dataset and generates the final batch output of dataloader
            # self.loaders[split] = data.DataLoader(dataset=self.datasets[split], batch_size=self.batch_size,
            self.loaders[split] = DataLoaderX(dataset=self.datasets[split], batch_size=self.batch_size,
                                              sampler=sampler, pin_memory=False, num_workers=0,
                                              collate_fn=self.datasets[split].collate_func, drop_last=False)

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()

    def get_vocab_size(self):
        return self.datasets['train'].get_vocab_size()

    def get_src_vocab_size(self):
        return self.datasets['train'].get_src_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    @property
    def src_vocab_size(self):
        return self.get_src_vocab_size()

    def get_vocab(self):
        return self.datasets['train'].get_vocab()

    def get_src_vocab(self):
        return self.datasets['train'].get_src_vocab()

    def get_seq_length(self):
        return self.datasets['train'].get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self, prefetched_num=None):
        return {split: self.loaders[split].sampler.state_dict(prefetched_num) for split in ['train', 'val', 'test']}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle):
        self.index_list = index_list
        self.shuffle = shuffle
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            raise StopIteration()
        if len(self._index_list) == 0:  # overflow when 0 samples
            return None
        elem = self._index_list[self.iter_counter]
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }

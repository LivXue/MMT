from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from CaptionModel import CaptionModel
from modules.sentence_tramsformer import *

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
               'are', 'am']


def repeat_tensors(n, x):
    if torch.is_tensor(x):
        x = x.unsqueeze(1)
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))
        x = x.reshape(x.shape[0] * n, *x.shape[2:])
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


class MMT(CaptionModel):
    """
    Model of "MMT: Image-guided Story Ending Generation with Multimodal Memory Transformer"
    """
    def __init__(self, opt):
        super(MMT, self).__init__()
        # self.core = UpDownCore(opt)
        self.vocab_size = opt.vocab_size
        print('tgt_vocab_size:', self.vocab_size)
        self.common_size = opt.common_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.img_pool_size = opt.img_pool_size
        self.img_conv_size = opt.img_conv_size

        self.bos_idx = getattr(opt, 'bos_idx', 0)
        self.eos_idx = getattr(opt, 'eos_idx', 0)
        self.pad_idx = getattr(opt, 'pad_idx', 0)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0

        if opt.use_bert:
            self.sentence_transformer = BERTTransformer(opt)
        else:
            self.sentence_transformer = SentenceTransformer(opt)
        self.sentence_generator = MTDEndingGenerator(opt)

        self.word_emb = nn.Embedding(self.vocab_size, opt.tgt_vocab_dim)
        word_emb = torch.load(opt.dec_emb_path)
        self.word_emb.weight = nn.Parameter(word_emb, requires_grad=True)

        self.img_conv_feature = nn.Sequential(nn.Linear(self.img_conv_size, self.common_size),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(self.drop_prob_lm),
                                              nn.LayerNorm(self.common_size))

        self.modal_fusion = MultiHeadAttention(self.common_size, self.common_size // 2, self.common_size // 2,
                                               opt.num_head, memory_size=opt.memory_size)

        self.logit = nn.Linear(opt.output_size, self.vocab_size, bias=False)

        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k, v in self.vocab.items() if v in bad_endings]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return [(weight.new_zeros(batch_size, self.common_size), weight.new_zeros(batch_size, self.common_size)),
                (weight.new_zeros(batch_size, self.common_size), weight.new_zeros(batch_size, self.common_size))]

    def _forward(self, img_conv_feature, conv_masks, seq, src1, src2, src3, src4):
        sent_word_features, sent_masks = self.sentence_transformer(src1, src2, src3, src4)

        sent1_word_feature, sent2_word_feature, sent3_word_feature, sent4_word_feature = sent_word_features
        sent1_mask, sent2_mask, sent3_mask, sent4_mask = sent_masks

        # Project to common feature space
        img_conv_feature = self.img_conv_feature(img_conv_feature)

        if conv_masks is not None:
            img_sent_mask = (~conv_masks).unsqueeze(1).repeat(1, src1.size()[1], 1)
        else:
            img_sent_mask = None

        sent1_word_fusion_feature, _ = self.modal_fusion(sent1_word_feature, img_conv_feature, img_conv_feature,
                                                         attention_mask=img_sent_mask)
        sent2_word_fusion_feature, _ = self.modal_fusion(sent2_word_feature, img_conv_feature, img_conv_feature,
                                                         attention_mask=img_sent_mask)
        sent3_word_fusion_feature, _ = self.modal_fusion(sent3_word_feature, img_conv_feature, img_conv_feature,
                                                         attention_mask=img_sent_mask)
        sent4_word_fusion_feature, _ = self.modal_fusion(sent4_word_feature, img_conv_feature, img_conv_feature,
                                                         attention_mask=img_sent_mask)

        img_sent1_mask = (~sent1_mask).unsqueeze(1).repeat(1, img_conv_feature.size()[1], 1)
        img_sent2_mask = (~sent2_mask).unsqueeze(1).repeat(1, img_conv_feature.size()[1], 1)
        img_sent3_mask = (~sent3_mask).unsqueeze(1).repeat(1, img_conv_feature.size()[1], 1)
        img_sent4_mask = (~sent4_mask).unsqueeze(1).repeat(1, img_conv_feature.size()[1], 1)
        img_fusion_sent1_feature, _ = self.modal_fusion(img_conv_feature, sent1_word_feature, sent1_word_feature,
                                                        attention_mask=img_sent1_mask)
        img_fusion_sent2_feature, _ = self.modal_fusion(img_conv_feature, sent2_word_feature, sent2_word_feature,
                                                        attention_mask=img_sent2_mask)
        img_fusion_sent3_feature, _ = self.modal_fusion(img_conv_feature, sent3_word_feature, sent3_word_feature,
                                                        attention_mask=img_sent3_mask)
        img_fusion_sent4_feature, _ = self.modal_fusion(img_conv_feature, sent4_word_feature, sent4_word_feature,
                                                        attention_mask=img_sent4_mask)

        sent_word_fusion_features = [sent1_word_fusion_feature, sent2_word_fusion_feature, sent3_word_fusion_feature,
                                     sent4_word_fusion_feature]
        img_fusion_sent_features = [img_fusion_sent1_feature, img_fusion_sent2_feature, img_fusion_sent3_feature,
                                    img_fusion_sent4_feature]


        sent_features = [torch.sum(sent_word_fusion_feature * mask.unsqueeze(-1), dim=1) /
                         mask.float().sum(dim=1).unsqueeze(-1) for sent_word_fusion_feature, mask in
                         zip(sent_word_fusion_features, sent_masks)]
        if conv_masks is not None:
            img_features = [torch.sum(img_fusion_sent_feature * conv_masks.unsqueeze(-1), dim=1) /
                            conv_masks.float().sum(dim=1, keepdims=True) for img_fusion_sent_feature in
                            img_fusion_sent_features]
        else:
            img_features = [img_fusion_sent_feature.mean(dim=1) for img_fusion_sent_feature in
                            img_fusion_sent_features]

        batch_size = img_conv_feature.size(0)

        if seq.ndim == 3:
            seq = seq.reshape(-1, seq.shape[2])

        outputs = img_conv_feature.new_zeros(batch_size, seq.size(1), self.vocab_size)

        state_features = None  # self.init_hidden(batch_size)
        for i in range(seq.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = img_conv_feature.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    word_idx = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    word_idx = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    word_idx.index_copy_(0, sample_ind,
                                         torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                word_idx = seq[:, i].clone()
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state_features = self.get_logprobs_state(word_idx, sent_features, img_features, state_features)

            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, word_idx, sent_features, img_feature, state_features, output_logsoftmax=True):

        pre_word_emb = self.word_emb(word_idx)

        output_feature, state_features = self.sentence_generator(pre_word_emb, sent_features, img_feature,
                                                                 state_features)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output_feature), dim=1)
        else:
            logprobs = self.logit(output_feature)

        return logprobs, state_features

    def _sample(self, img_conv_feature, conv_masks, src1, src2, src3, src4, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', True)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(img_conv_feature, opt)
        if group_size > 1:
            return self._diverse_sample(img_conv_feature, opt)

        batch_size = img_conv_feature.size(0)

        sent_word_features, sent_masks = self.sentence_transformer(src1, src2, src3, src4)

        sent1_word_feature, sent2_word_feature, sent3_word_feature, sent4_word_feature = sent_word_features
        sent1_mask, sent2_mask, sent3_mask, sent4_mask = sent_masks

        # Project to common feature space
        img_conv_feature = self.img_conv_feature(img_conv_feature)

        if conv_masks is not None:
            img_sent_mask = (~conv_masks).unsqueeze(1).repeat(1, src1.size()[1], 1)
        else:
            img_sent_mask = None

        sent1_word_fusion_feature, _ = self.modal_fusion(sent1_word_feature, img_conv_feature, img_conv_feature,
                                                         attention_mask=img_sent_mask)
        sent2_word_fusion_feature, _ = self.modal_fusion(sent2_word_feature, img_conv_feature, img_conv_feature,
                                                         attention_mask=img_sent_mask)
        sent3_word_fusion_feature, _ = self.modal_fusion(sent3_word_feature, img_conv_feature, img_conv_feature,
                                                         attention_mask=img_sent_mask)
        sent4_word_fusion_feature, _ = self.modal_fusion(sent4_word_feature, img_conv_feature, img_conv_feature,
                                                         attention_mask=img_sent_mask)

        img_sent1_mask = (~sent1_mask).unsqueeze(1).repeat(1, img_conv_feature.size()[1], 1)
        img_sent2_mask = (~sent2_mask).unsqueeze(1).repeat(1, img_conv_feature.size()[1], 1)
        img_sent3_mask = (~sent3_mask).unsqueeze(1).repeat(1, img_conv_feature.size()[1], 1)
        img_sent4_mask = (~sent4_mask).unsqueeze(1).repeat(1, img_conv_feature.size()[1], 1)
        img_fusion_sent1_feature, _ = self.modal_fusion(img_conv_feature, sent1_word_feature, sent1_word_feature,
                                                        attention_mask=img_sent1_mask)
        img_fusion_sent2_feature, _ = self.modal_fusion(img_conv_feature, sent2_word_feature, sent2_word_feature,
                                                        attention_mask=img_sent2_mask)
        img_fusion_sent3_feature, _ = self.modal_fusion(img_conv_feature, sent3_word_feature, sent3_word_feature,
                                                        attention_mask=img_sent3_mask)
        img_fusion_sent4_feature, _ = self.modal_fusion(img_conv_feature, sent4_word_feature, sent4_word_feature,
                                                        attention_mask=img_sent4_mask)

        sent_word_fusion_features = [sent1_word_fusion_feature, sent2_word_fusion_feature, sent3_word_fusion_feature,
                                     sent4_word_fusion_feature]
        img_fusion_sent_features = [img_fusion_sent1_feature, img_fusion_sent2_feature, img_fusion_sent3_feature,
                                    img_fusion_sent4_feature]

        sent_features = [torch.sum(sent_word_fusion_feature * mask.unsqueeze(-1), dim=1) /
                         mask.float().sum(dim=1, keepdims=True) for sent_word_fusion_feature, mask in
                         zip(sent_word_fusion_features, sent_masks)]
        if conv_masks is not None:
            img_features = [torch.sum(img_fusion_sent_feature * conv_masks.unsqueeze(-1), dim=1) /
                            conv_masks.float().sum(dim=1, keepdims=True) for img_fusion_sent_feature in
                            img_fusion_sent_features]
        else:
            img_features = [img_fusion_sent_feature.mean(dim=1) for img_fusion_sent_feature in
                            img_fusion_sent_features]

        if sample_n > 1:
            sent_features, img_features = repeat_tensors(sample_n, [sent_features, img_features])

        trigrams = []

        seq = img_conv_feature.new_full((batch_size * sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = img_conv_feature.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size)
        state_features = None  # self.init_hidden(batch_size)

        for t in range(self.seq_length + 1):
            if t == 0:
                word_idx = img_conv_feature.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state_features = self.get_logprobs_state(word_idx, sent_features, img_features, state_features,
                                                               output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            if block_trigrams and t >= 3:
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:
                        trigrams.append({prev_two: [current]})
                    elif t > 3:
                        if prev_two in trigrams[i]:
                            trigrams[i][prev_two].append(current)
                        else:
                            trigrams[i][prev_two] = [current]
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                alpha = 2.0
                logprobs = logprobs + (mask * -0.693 * alpha)

            if t == self.seq_length:
                break
            word_idx, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            if t == 0:
                unfinished = word_idx != self.eos_idx
            else:
                word_idx[~unfinished] = self.pad_idx
                logprobs = logprobs * unfinished.unsqueeze(1)
                unfinished = unfinished * (word_idx != self.eos_idx)
            seq[:, t] = word_idx
            seqLogprobs[:, t] = logprobs
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

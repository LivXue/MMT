import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


class MultiHeadAttention(nn.Module):
    """
    multi-head attention layer with MEMORY module
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, use_res=True, memory_size=0):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.use_res = use_res

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.memory_size = memory_size
        if memory_size > 0:
            memory_k = torch.Tensor(memory_size, h * d_k)
            nn.init.xavier_uniform_(memory_k)
            memory_v = torch.Tensor(memory_size, h * d_v)
            nn.init.xavier_uniform_(memory_v)
            self.m_k = nn.Parameter(memory_k.view(1, memory_size, self.h, self.d_k).permute(0, 2, 3, 1)) # (1, h, d_k, nm)
            self.m_v = nn.Parameter(memory_v.view(1, memory_size, self.h, self.d_v).permute(0, 2, 1, 3)) # (1, h, nm, d_v)
            self.m_weight = torch.ones(1, 1, memory_size).cuda()
            self.m_mask = torch.zeros(1, 1, memory_size, dtype=torch.bool).cuda()
            self.m_prev = torch.zeros(1, 1, memory_size).cuda()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, prev_att=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if self.memory_size > 0:
            k = torch.cat((k, self.m_k.repeat(b_s, 1, 1, 1)), dim=3)
            v = torch.cat((v, self.m_v.repeat(b_s, 1, 1, 1)), dim=2)
            if attention_weights is not None:
                attention_weights = torch.cat((attention_weights, self.m_weight.repeat(b_s, nq, 1)), dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, self.m_mask.repeat(b_s, nq, 1)), dim=-1)
            if prev_att is not None:
                prev_att = torch.cat((prev_att, self.m_prev.repeat(b_s, nq, 1)), dim=-1)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if prev_att is not None:
            att = att + prev_att
        else:
            att = att
        if attention_mask is not None:
            r_att = att.masked_fill(attention_mask.unsqueeze(1), -np.inf)
            # mask before softmax, so exp(-inf) will be 0
        else:
            r_att = att
        att = torch.softmax(r_att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        out = self.dropout(out)
        if self.use_res:
            out = self.layer_norm(queries + out)
        else:
            out = self.layer_norm(out)
        return out, r_att

    def get_q_k(self, feature):
        b_s, n = feature.shape[:2]
        q = self.fc_q(feature).view(b_s, n, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, n, d_k)
        k = self.fc_k(feature).view(b_s, n, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n)

        return q, k


class FourSentenceSelfAttetion(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(FourSentenceSelfAttetion, self).__init__()
        self.sent_self_attention = MultiHeadAttention(d_model, d_k, d_v, h, dropout)

    def forward(self, sent_word_features, sent_att_masks):
        sent1_word_feature, sent2_word_feature, sent3_word_feature, sent4_word_feature = sent_word_features
        sent1_att_mask, sent2_att_mask, sent3_att_mask, sent4_att_mask = sent_att_masks

        sent1_word_feature, _ = self.sent_self_attention(sent1_word_feature, sent1_word_feature, sent1_word_feature,
                                                         attention_mask=sent1_att_mask)
        sent2_word_feature, _ = self.sent_self_attention(sent2_word_feature, sent2_word_feature, sent2_word_feature,
                                                         attention_mask=sent2_att_mask)
        sent3_word_feature, _ = self.sent_self_attention(sent3_word_feature, sent3_word_feature, sent3_word_feature,
                                                         attention_mask=sent3_att_mask)
        sent4_word_feature, _ = self.sent_self_attention(sent4_word_feature, sent4_word_feature, sent4_word_feature,
                                                         attention_mask=sent4_att_mask)
        return [sent1_word_feature, sent2_word_feature, sent3_word_feature, sent4_word_feature]


class FourSentenceLinearLayer(nn.Module):
    def __init__(self, d_in=300, d_out=1024, use_relu=True, use_res=False, use_layer_norm=False):
        super(FourSentenceLinearLayer, self).__init__()
        self.linear_layer = nn.Linear(d_in, d_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_out)

        self.use_relu = use_relu
        self.use_res = use_res
        self.use_layer_norm = use_layer_norm

    def forward(self, sent_word_features):
        sent1_word_feature, sent2_word_feature, sent3_word_feature, sent4_word_feature = sent_word_features

        sent1_word_feature_out = self.linear_layer(sent1_word_feature)
        sent2_word_feature_out = self.linear_layer(sent2_word_feature)
        sent3_word_feature_out = self.linear_layer(sent3_word_feature)
        sent4_word_feature_out = self.linear_layer(sent4_word_feature)

        if self.use_relu:
            sent1_word_feature_out = self.relu(sent1_word_feature_out)
            sent2_word_feature_out = self.relu(sent2_word_feature_out)
            sent3_word_feature_out = self.relu(sent3_word_feature_out)
            sent4_word_feature_out = self.relu(sent4_word_feature_out)

        if self.use_res:
            sent1_word_feature_out = sent1_word_feature + sent1_word_feature_out
            sent2_word_feature_out = sent2_word_feature + sent2_word_feature_out
            sent3_word_feature_out = sent3_word_feature + sent3_word_feature_out
            sent4_word_feature_out = sent4_word_feature + sent4_word_feature_out

        if self.use_layer_norm:
            sent1_word_feature_out = self.layer_norm(sent1_word_feature_out)
            sent2_word_feature_out = self.layer_norm(sent2_word_feature_out)
            sent3_word_feature_out = self.layer_norm(sent3_word_feature_out)
            sent4_word_feature_out = self.layer_norm(sent4_word_feature_out)

        return [sent1_word_feature_out, sent2_word_feature_out, sent3_word_feature_out, sent4_word_feature_out]


class FourSentenceTransformerLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, use_position_emb=False, max_seq_len=40):
        super(FourSentenceTransformerLayer, self).__init__()
        self.self_att = FourSentenceSelfAttetion(d_model, d_k, d_v, h, dropout)
        self.feed_forward = FourSentenceLinearLayer(d_model, d_model, use_relu=True, use_res=True, use_layer_norm=True)

        self.use_position_emb = use_position_emb
        if self.use_position_emb:
            tmp = torch.Tensor(max_seq_len, d_model)
            nn.init.xavier_normal_(tmp)
            self.position_emb = nn.Parameter(tmp.unsqueeze(0))

    def forward(self, sent_word_features, sent_att_masks):
        if self.use_position_emb:
            sent_word_features = [sent_word_feature + self.position_emb for sent_word_feature in sent_word_features]

        sent_word_features = self.self_att(sent_word_features, sent_att_masks)
        sent_word_features = self.feed_forward(sent_word_features)

        return sent_word_features


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, use_position_emb=False, max_seq_len=41):
        super(TransformerLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(d_model)

        self.use_position_emb = use_position_emb
        if self.use_position_emb:
            tmp = torch.Tensor(max_seq_len, d_model)
            nn.init.xavier_normal_(tmp)
            self.position_emb = nn.Parameter(tmp.unsqueeze(0))

    def forward(self, input, mask=None, only_last=False):
        if self.use_position_emb:
            input = input + self.position_emb[:, -input.size()[1]:]

        if only_last:
            feature, _ = self.self_att(input[:, -1].unsqueeze(1), input, input, mask)
            feature = feature.squeeze(1)
        else:
            feature, _ = self.self_att(input, input, input, mask)
        feature = self.feed_forward(feature)
        feature = self.relu(feature)
        if only_last:
            feature = input[:, -1] + feature
        else:
            feature = input + feature
        feature = self.layer_norm(feature)

        return feature


class SentenceTransformer(nn.Module):
    def __init__(self, opt):
        super(SentenceTransformer, self).__init__()
        self.opt = opt

        src_embed = torch.load(opt.enc_emb_path)
        vocab = src_embed.size()[0]
        self.src_word_emb = nn.Embedding(vocab, opt.src_vocab_dim, padding_idx=0)
        self.src_word_emb.weight = nn.Parameter(src_embed)

        self.num_head = getattr(opt, 'num_head', 8)

        self.input_linear = FourSentenceLinearLayer(opt.src_vocab_dim, opt.common_size)
        self.transformer_layer1 = FourSentenceTransformerLayer(opt.common_size, opt.common_size // 2,
                                                               opt.common_size // 2, self.num_head,
                                                               dropout=opt.drop_prob_lm, use_position_emb=True,
                                                               max_seq_len=opt.seq_length)
        self.transformer_layer2 = FourSentenceTransformerLayer(opt.common_size, opt.common_size // 2,
                                                               opt.common_size // 2, self.num_head,
                                                               dropout=opt.drop_prob_lm, use_position_emb=True,
                                                               max_seq_len=opt.seq_length)

    def forward(self, sent1, sent2, sent3, sent4):
        sent1_emb = self.src_word_emb(sent1)
        sent2_emb = self.src_word_emb(sent2)
        sent3_emb = self.src_word_emb(sent3)
        sent4_emb = self.src_word_emb(sent4)
        sent_word_embs = [sent1_emb, sent2_emb, sent3_emb, sent4_emb]

        sent1_mask = (sent1 != 0)
        sent2_mask = (sent2 != 0)
        sent3_mask = (sent3 != 0)
        sent4_mask = (sent4 != 0)
        sent_masks = [sent1_mask, sent2_mask, sent3_mask, sent4_mask]

        sent1_att_mask = ~sent1_mask.unsqueeze(1)
        sent2_att_mask = ~sent2_mask.unsqueeze(1)
        sent3_att_mask = ~sent3_mask.unsqueeze(1)
        sent4_att_mask = ~sent4_mask.unsqueeze(1)
        sent_att_masks = [sent1_att_mask, sent2_att_mask, sent3_att_mask, sent4_att_mask]

        sent_word_features = self.input_linear(sent_word_embs)
        sent_word_features = self.transformer_layer1(sent_word_features, sent_att_masks)
        sent_word_features = self.transformer_layer2(sent_word_features, sent_att_masks)

        return sent_word_features, sent_masks


class BERTTransformer(nn.Module):
    def __init__(self, opt):
        super(BERTTransformer, self).__init__()
        self.opt = opt
        self.bert_model = BertModel.from_pretrained(opt.bert_architecture)
        self.fine_tuning = False

        self.output_linear = nn.Linear(768, opt.common_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(opt.common_size)

    def forward(self, sent1, sent2, sent3, sent4):
        sent1_mask = (sent1 != 0)
        sent2_mask = (sent2 != 0)
        sent3_mask = (sent3 != 0)
        sent4_mask = (sent4 != 0)
        sent_masks = [sent1_mask, sent2_mask, sent3_mask, sent4_mask]

        sent1_feature = self.bert_model(input_ids=sent1, attention_mask=sent1_mask)[0]
        sent2_feature = self.bert_model(input_ids=sent2, attention_mask=sent2_mask)[0]
        sent3_feature = self.bert_model(input_ids=sent3, attention_mask=sent3_mask)[0]
        sent4_feature = self.bert_model(input_ids=sent4, attention_mask=sent4_mask)[0]

        if self.fine_tuning is False:
            sent1_feature = sent1_feature.detach()
            sent2_feature = sent2_feature.detach()
            sent3_feature = sent3_feature.detach()
            sent4_feature = sent4_feature.detach()

        sent1_feature = self.layer_norm(self.dropout(self.output_linear(sent1_feature)))
        sent2_feature = self.layer_norm(self.dropout(self.output_linear(sent2_feature)))
        sent3_feature = self.layer_norm(self.dropout(self.output_linear(sent3_feature)))
        sent4_feature = self.layer_norm(self.dropout(self.output_linear(sent4_feature)))
        sent_word_features = [sent1_feature, sent2_feature, sent3_feature, sent4_feature]

        return sent_word_features, sent_masks


class MTDEndingGenerator(nn.Module):
    def __init__(self, opt):
        super(MTDEndingGenerator, self).__init__()
        self.opt = opt
        self.output_size = getattr(opt, 'output_size', 1024)
        self.num_head = getattr(opt, 'num_head', 8)

        self.input_word_linear = nn.Linear(opt.tgt_vocab_dim, 512)
        self.relu = nn.ReLU(inplace=True)
        self.input_linear = nn.Linear(512 + 2 * opt.common_size, opt.common_size)
        self.transformer_layer1 = TransformerLayer(opt.common_size, opt.common_size // 2, opt.common_size // 2,
                                                   self.num_head, dropout=opt.drop_prob_lm)
        self.transformer_layer2 = TransformerLayer(opt.common_size, opt.common_size // 2, opt.common_size // 2,
                                                   self.num_head, dropout=opt.drop_prob_lm)
        self.output_linear = nn.Linear(opt.common_size, self.output_size)
        self.dropout = nn.Dropout(opt.drop_prob_lm)
        self.layer_norm = nn.LayerNorm(self.output_size)

        self.gen_position_emb_layer1 = nn.Parameter(self.init_tensor(41, opt.common_size).unsqueeze(0))
        self.gen_position_emb_layer2 = nn.Parameter(self.init_tensor(41, opt.common_size).unsqueeze(0))
        self.sent_img_position_emb_layer1 = nn.Parameter(self.init_tensor(8, opt.common_size).unsqueeze(0))
        self.sent_img_position_emb_layer2 = nn.Parameter(self.init_tensor(8, opt.common_size).unsqueeze(0))

    def init_tensor(self, dim1, dim2):
        tmp = torch.Tensor(dim1, dim2)
        nn.init.xavier_uniform_(tmp)
        return tmp

    def add_position_emb(self, features, gen_position_emb, sent_img_position_emb, seq_len):
        position_emb = torch.cat((sent_img_position_emb, gen_position_emb[:, -seq_len:]), dim=1)
        return features + position_emb

    def forward(self, pre_word_emb, sent_features, img_features, state_features=None):
        sent_feature = (sent_features[0] + sent_features[1] + sent_features[2] + sent_features[3]) / 4
        img_feature = (img_features[0] + img_features[1] + img_features[2] + img_features[3]) / 4

        pre_word_feature = self.input_word_linear(pre_word_emb)
        pre_word_feature = self.relu(pre_word_feature)

        current_feature = torch.cat((pre_word_feature, sent_feature, img_feature), dim=1)
        current_feature = self.input_linear(current_feature)
        current_feature = self.relu(current_feature)

        if state_features is not None:
            state_features = torch.cat((state_features, current_feature.unsqueeze(1)), dim=1).contiguous()
        else:
            concat_sent_feature = torch.cat([sent_feature.unsqueeze(1) for sent_feature in sent_features], dim=1)
            concat_img_feature = torch.cat([img_feature.unsqueeze(1) for img_feature in img_features], dim=1)
            state_features = torch.cat((concat_sent_feature, concat_img_feature, current_feature.unsqueeze(1)), dim=1)

        batch_size, seq_len = state_features.size()[0], state_features.size()[1] - 8

        input_feature = self.add_position_emb(state_features, self.gen_position_emb_layer1,
                                              self.sent_img_position_emb_layer1, seq_len)
        hidden_features = self.transformer_layer1(input_feature)#, att_masks)
        hidden_features = self.add_position_emb(hidden_features, self.gen_position_emb_layer2,
                                                self.sent_img_position_emb_layer2, seq_len)
        hidden_features = self.transformer_layer2(hidden_features, only_last=True)

        output_feature = self.output_linear(hidden_features)
        output_feature = self.dropout(self.relu(output_feature))
        output_feature = self.layer_norm(output_feature)

        return output_feature, state_features

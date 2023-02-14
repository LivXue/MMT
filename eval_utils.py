from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import utils
from metrics import show_all_scores


def eval_split(model, lw_model, loader, eval_kwargs=None):
    if eval_kwargs is None:
        eval_kwargs = {}
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    running_loss = 0.0
    predictions = []
    gts = []
    res = []
    for data in tqdm(loader.loaders[split]):
        tmp = [data['fc_feats'], data['conv_feats'], data['labels'], data['masks'], data['conv_masks'], data['src1'],
               data['src2'], data['src3'], data['src4']]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, conv_feats, labels, masks, conv_masks, src1, src2, src3, src4 = tmp
        if labels is not None and verbose_loss:
            # Forward the model to get loss
            with torch.no_grad():
                loss = lw_model(conv_feats, conv_masks, labels, masks, src1, src2, src3, src4).item()
            running_loss += loss

        # Forward the model to also get generated samples for each image
        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            seq, seq_logprobs = model(conv_feats, conv_masks, src1, src2, src3, src4, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq > 0).float().sum(1) + 1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq > 0).float().sum(1) + 1)

        # Conduct beam search
        sents = utils.decode_sequence(model.vocab, seq)

        if lang_eval:
            gts.extend(utils.decode_sequence(model.vocab, np.squeeze(labels)[:, 1:]))
            res.extend(sents)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(),
                     'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # Dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + \
                      '" vis/imgs/img' + str(len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

    lang_stats = None
    if lang_eval:
        bleu_results, meteor_results, cider_results, rouge_results, rsum = show_all_scores(gts, res, n=4)
        lang_stats = {'BLEU 1': bleu_results['BLEU 1'], 'BLEU 2': bleu_results['BLEU 2'],
                      'BLEU 3': bleu_results['BLEU 3'], 'BLEU 4': bleu_results['BLEU 4'],
                      'METEOR': meteor_results['METEOR'], 'CIDEr': cider_results['CIDEr'],
                      'ROUGE-L': rouge_results['ROUGE-L'], 'rSUM': rsum}

    # Switch back to training mode
    model.train()
    return running_loss / len(loader.loaders[split]), predictions, lang_stats

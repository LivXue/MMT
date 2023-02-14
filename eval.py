from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os

import torch
import json

import utils
import utils.opts as opts
from model import MMT
from dataloader.dataloader import DataLoader
from eval_utils import eval_split
import modules.losses as losses
from modules.loss_wrapper import LossWrapper


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--model', type=str, default='./log/VIST-E/log_bert/model-best.pth', help='path to model to evaluate')
    parser.add_argument('--infos_path', type=str, default='./log/VIST-E/log_bert/infos_6-best.pkl', help='path to infos to evaluate')
    opts.add_eval_options(parser)
    opt = parser.parse_args()

    # Load infos
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    # Override and collect parameters
    replace = ['input_fc_dir', 'input_conv_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
    ignore = ['start_from']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if k not in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

    vocab = infos['vocab']  # ix -> word mapping

    # Setup the model
    opt.vocab = vocab
    model = MMT(opt)
    model.load_state_dict(torch.load(opt.model))
    lw_model = LossWrapper(model, opt)
    lw_model.cuda()
    lw_model.eval()
    crit = losses.LanguageModelCriterion()

    # Create the Data Loader instance
    loader = DataLoader(opt)

    # When eval using provided pretrained model, the vocab may be different from what you have in your data_res.json
    # So make sure to use the vocab in infos file.
    loader.datasets['test'].ix_to_word = infos['vocab']

    # Set sample options
    opt.dataset = opt.input_json
    loss, split_predictions, lang_stats = eval_split(model, lw_model, loader, vars(opt))

    path_gen = './data/gen/gen_ending.txt'
    file_txt = open(path_gen, 'w')
    end_gen = []
    for story in split_predictions:
        story_gen = story['caption']
        #story_gen = story_gen.replace("[ ", "[").replace(" ]", "]")
        end_gen.append(story_gen)
        file_txt.write(story_gen)
        file_txt.write('\r\n')
    file_txt.close()

    print('Evaluation loss: ', loss)
    if lang_stats:
        print(lang_stats)

    if opt.dump_json == 1:
        # Dump the json
        json.dump(split_predictions, open('vis/vis.json', 'w'))

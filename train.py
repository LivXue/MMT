from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from collections import defaultdict

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from utils import opts as opts
from model import MMT
from dataloader.dataloader import DataLoader
from eval_utils import eval_split
from modules.loss_wrapper import LossWrapper


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    since = time.time()

    print('...Data loading is beginning...')
    loader = DataLoader(opt)
    print('...Data loading is completed...')

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt,
                                                                    checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))

    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    opt.vocab = loader.get_vocab()
    opt.src_vocab = loader.get_src_vocab()
    model = MMT(opt).cuda()
    del opt.vocab
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
        print('path: ', os.path.join(opt.start_from, 'model.pth'))
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    lw_model = LossWrapper(model, opt)
    dp_model = torch.nn.DataParallel(model)
    dp_model = dp_model.module
    dp_lw_model = torch.nn.DataParallel(lw_model)
    dp_lw_model = dp_lw_model.module

    optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        print('start from is not none')
        print('optimizer_path: ', opt.start_from, "optimizer.pth")
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    iteration = infos['iter']
    if 'iterators' in infos:
        infos['loader_state_dict'] = {
            split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in
            ['train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict'])

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    else:
        best_val_score = None

    dp_lw_model.train()
    early_stop = 0
    opt.current_lr = opt.learning_rate

    for epoch in range(infos['epoch'], opt.max_epochs):
        print('Epoch {}/{}'.format(epoch + 1, opt.max_epochs))
        print('-' * 20)
        if epoch >= opt.learning_rate_decay_start >= 0:
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            utils.set_lr(optimizer, opt.current_lr)
        if epoch >= opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob

        if epoch >= opt.bert_finetuning >= 0 and opt.use_bert:
            model.sentence_transformer.fine_tuning = True

        if opt.use_warmup and epoch < opt.warmup:
            opt.current_lr = opt.learning_rate * (iteration + 1) / opt.warmup
            utils.set_lr(optimizer, opt.current_lr)

        running_loss = 0.0
        # Load data from train split (0)
        for data in tqdm(loader.loaders['train']):
            iteration += 1
            tmp = [data['fc_feats'], data['conv_feats'], data['labels'], data['masks'], data['conv_masks'], data['src1'],
                data['src2'], data['src3'], data['src4']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, conv_feats, labels, masks, conv_masks, src1, src2, src3, src4 = tmp

            optimizer.zero_grad()
            loss = dp_lw_model(conv_feats, conv_masks, labels, masks, src1, src2, src3, src4)
            loss.backward()
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' % opt.grad_clip_mode)(model.parameters(), opt.grad_clip_value)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader.loaders['train'])
        print("Train loss: {}".format(epoch_loss))

        tb_summary_writer.add_scalar('train_loss', epoch_loss, epoch)
        tb_summary_writer.add_scalar('learning_rate', opt.current_lr, epoch)
        tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, epoch)

        histories['loss_history'][epoch] = epoch_loss
        histories['lr_history'][epoch] = opt.current_lr
        histories['ss_prob_history'][epoch] = model.ss_prob

        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['loader_state_dict'] = loader.state_dict()

        # TODO:
        # Use split=test for easy start and use split=val for strict experiments
        eval_kwargs = {'split': 'test', 'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        val_loss, predictions, lang_stats = eval_split(dp_model, lw_model, loader, eval_kwargs)

        tb_summary_writer.add_scalar('validation loss', val_loss, epoch)
        if lang_stats is not None:
            for k, v in lang_stats.items():
                tb_summary_writer.add_scalar(k, v, epoch)
        histories['val_result_history'][epoch] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

        if opt.language_eval == 1:
            current_score = lang_stats['rSUM']
        else:
            current_score = - val_loss
        print("Validate loss: {}".format(val_loss))

        if best_val_score is None or current_score > best_val_score:
            best_val_score = current_score
            utils.save_checkpoint(opt, model, infos, optimizer, append='best')
            infos['best_val_score'] = best_val_score
            early_stop = 0
        else:
            early_stop += 1

        utils.save_checkpoint(opt, model, infos, optimizer, histories)

        if early_stop == opt.early_stop:
            print("Early stop!")
            break

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    opt = opts.parse_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    seed = 104
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Random seed = {}".format(seed))

    train(opt)

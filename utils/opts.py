from __future__ import print_function
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--dataset', type=str, default='VIST',
                        help='dataset: VIST / LSMDC')
    parser.add_argument('--data_in_memory', action='store_true',
                        help='True if we want to save the features in memory')
    parser.add_argument('--start_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'model.pth'         : weights""")
    parser.add_argument('--src_vocab_dim', type=int, default=300,
                        help="dimension for source word embeddings")
    parser.add_argument('--tgt_vocab_dim', type=int, default=300,
                        help="dimension for target word embeddings")

    # Model settings
    parser.add_argument('--common_size', type=int, default=1024,
                        help='Multimodal common feature size')
    parser.add_argument('--output_size', type=int, default=1024,
                        help='Language generator output size')
    parser.add_argument('--img_pool_size', type=int, default=2048,
                        help='Image pooled feature size')
    parser.add_argument('--img_conv_size', type=int, default=2048,
                        help='Last dimensionality of image convolutional feature')
    parser.add_argument('--num_head', type=int, default=4,
                        help='Number of heads in multi-head attention')
    parser.add_argument('--memory_size', type=int, default=40,
                        help='Size of generator memory')

    parser.add_argument('--flat_glimpses', type=int, default=1)

    # feature manipulation
    parser.add_argument('--norm_conv_feat', type=int, default=0,
                        help='If normalize convention features')
    parser.add_argument('--use_fc', type=bool, default=False,
                        help='If use fc features')
    parser.add_argument('--use_conv', type=bool, default=True,
                        help='If use convolutional features')
    parser.add_argument('--use_box', type=bool, default=False,
                        help='If use box features')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                        help='If use box, do we normalize box feature')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--grad_clip_mode', type=str, default='value',
                        help='value or norm')
    parser.add_argument('--grad_clip_value', type=float, default=0.5,#0.1,
                        help='clip gradients at this value/max_norm, 0 means no clipping')
    parser.add_argument('--drop_prob_lm', type=float, default=0.3,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                        help='After what epoch do we start finetuning the CNN? '
                             '(-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=1,
                        help='number of captions to sample for each image during training. '
                             'Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam|adamw')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=5,
                        help='at what epoch to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                        help='every how many epoches thereafter to drop learning rate?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5,
                        help='rate of weight decaying')
    parser.add_argument('--optim_alpha', type=float, default=0.8,#0.5,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.99,#0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop number')
    parser.add_argument('--warmup', type=int, default=2,
                        help='warm up number')
    parser.add_argument('--use_warmup', action='store_true',
                        help='warm up the learing rate?')
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                        help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                        help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    # BERT settings
    parser.add_argument('--use_bert', type=bool, default=True,
                        help='If use BERT as the encoder.')
    parser.add_argument('--bert_architecture', type=str, default="bert-base-uncased",
                        help='BERT architecture.')
    parser.add_argument('--bert_finetuning', type=int, default=5,
                        help='Beginning epoch of finetuning, -1 means never.')

    # Evaluation
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU & CIDEr & METEOR & ROUGE_L?')
    parser.add_argument('--load_best_score', type=int, default=1,
                        help='Do we load previous best score when resuming training.')

    # misc
    parser.add_argument('--id', type=str, default='6',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')

    # Used for self critical
    parser.add_argument('--sc_sample_method', type=str, default='greedy',
                        help='')
    parser.add_argument('--sc_beam_size', type=int, default=1,
                        help='')

    args = parser.parse_args()

    # Paths for the dataset
    args.input_json = './data/' + args.dataset + '/data_res.json'
    args.input_fc_dir = ''
    args.input_conv_dir = './data/' + args.dataset + '/image_features'
    args.conv_ext = '.npz' if args.dataset == 'VIST' else '.npy'
    args.input_box_dir = ''
    args.input_label_h5 = './data/' + args.dataset + '/data_tgt_label.h5'
    args.input_src_h5 = './data/' + args.dataset + '/data_src_label.h5'
    args.input_adj_h5 = './data/' + args.dataset + '/data_adj_label.h5'
    args.enc_emb_path = './data/' + args.dataset + '/embedding/embedding_enc.pt'
    args.dec_emb_path = './data/' + args.dataset + '/embedding/embedding_dec.pt'

    # Maximum length of sentences
    args.seq_length = 40 if args.dataset == 'VIST' else 20

    # Check if args are valid
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.sc_beam_size > 0, "beam_size should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "load_best_score should be 0 or 1"

    # default value for start_from and checkpoint_path
    args.checkpoint_path = 'log/' + args.dataset + '/log_%s' % args.id
    args.start_from = args.start_from  # or args.checkpoint_path

    # Deal with feature things before anything
    if args.use_box:
        args.conv_feat_size = args.conv_feat_size + 5

    return args


def add_eval_options(parser):
    # Basic options
    parser.add_argument('--batch_size', type=int, default=0,
                        help='if > 0 then overrule, otherwise load from checkpoint.')
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1=yes, 0=no)? BLEU & CIDEr & METEOR & ROUGE-L.')
    parser.add_argument('--dump_images', type=int, default=0,
                        help='Dump images into vis/imgs folder for vis? (1=yes, 0=no)')
    parser.add_argument('--dump_json', type=int, default=0,
                        help='Dump json with predictions into vis folder? (1=yes, 0=no)')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes, 0=no)')
    parser.add_argument('--input_fc_dir', type=str, default='',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_conv_dir', type=str, default='./data/image_features',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_box_dir', type=str, default='',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_label_h5', type=str, default='',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_json', type=str, default='',
                        help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
    parser.add_argument('--split', type=str, default='test',
                        help='if running on MSCOCO images, which split to use: val|test|train')
    # misc
    parser.add_argument('--id', type=str, default='',
                        help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
    parser.add_argument('--verbose_loss', type=int, default=1,
                        help='If calculate loss using ground truth during evaluation')

    # For evaluation on a folder of images:
    parser.add_argument('--image_root', type=str, default='',
                        help='In case the image paths have to be preprended with a root path to an image folder')

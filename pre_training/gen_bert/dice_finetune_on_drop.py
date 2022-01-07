"""A Transformer with a BERT encoder and BERT decoder with extensive weight tying."""
# In each decoder layer, the self attention params are also used for source attention, 
# thereby allowing us to use BERT as a decoder as well.
# Most of the code is taken from HuggingFace's repo.

from __future__ import absolute_import

import argparse
import logging
import os, sys, random, jsonlines, shutil, time, re
import ujson as json
from scipy.special import softmax
from io import open
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from dice_modeling import BertTransformer, BertConfig, BertCtxClassifier
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer

from create_examples_n_features import DropExample, DropFeatures, read_file, write_file, split_digits
from squad_utils import exact_match_score, get_final_text
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1
from dice import DiceEmbedding

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
CONTEXT_TOKEN = '[CTX]'
CONTEXT_END = '[/CTX]'

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

START_TOK, END_TOK, SPAN_SEP, IGNORE_IDX, MAX_SPANS = '@', '\\', ';', 0, 6

logger.info("** torch.cuda.is_available ** : %s", str(torch.cuda.is_available()))

LMInputFeatures = namedtuple("LMInputFeatures", "input_ids input_mask lm_label_ids")


class MLMDataset(TensorDataset):
    def __init__(self, training_path, epoch=1, tokenizer=None, num_data_epochs=1):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.jsonl"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.jsonl"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        self.seq_len = seq_len = metrics['max_seq_len']
        
        input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=IGNORE_IDX) 
        # ignore index == 0
        logging.info(f"Loading MLM examples for epoch {epoch}")
        with jsonlines.open(data_file, 'r') as reader:
            for i, example in enumerate(tqdm(reader.iter(), total=num_samples, desc="MLM examples")):
                features = self.convert_example_to_features(example)
                input_ids[i] = features.input_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids

        assert i == num_samples - 1  # Assert that the sample count metric was true
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.lm_label_ids = lm_label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item]).long(),
                torch.tensor(self.input_masks[item]).long(),
                torch.tensor(self.lm_label_ids[item]).long())
    
    def convert_example_to_features(self, example):
        tokens = example["tokens"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
        max_seq_length = self.seq_len
        
        assert len(tokens) <= max_seq_length
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=IGNORE_IDX)
        lm_label_array[masked_lm_positions] = masked_label_ids

        features = LMInputFeatures(input_ids=input_array, input_mask=mask_array,
                                   lm_label_ids=lm_label_array)
        return features


ModelFeatures = namedtuple("ModelFeatures", "example_id input_ids input_mask segment_ids label_ids head_type q_spans p_spans \
                            numbers_in_input number_indices ctx_indices ctx_labels digit_pos digit_ent_indices digit_type_indices")


class DropDataset(TensorDataset):
    def __init__(self, args, split='train'):
        self.args = args

        logging.info(f"Loading {split} examples and features.")
        direc = args.examples_n_features_dir
        logging.info(f"Reading {args.surface_form} examples and features")

        if split == 'train':
            if (args.surface_form == "dice" and args.percent == "all") or args.surface_form == "ctx_only":  # dice and ctx_only
                example_path = direc + '/train_{}_examples.pkl'.format("local_ctx")
                feature_path = direc + '/train_{}_features.pkl'.format("local_ctx")
            elif args.surface_form in ["diceloss"] and args.percent == "all":  # diceembed
                example_path = direc + '/train_{}_examples.pkl'.format("dataset")
                feature_path = direc + '/train_{}_features.pkl'.format("dataset")
            elif (args.surface_form in ["numeric", "textual"] or args.surface_form in ["dicemlp", "pretrain_ctx", "digitctx", "attnmask"]) and args.percent == "all":
                example_path = direc + '/train_{}_examples.pkl'.format(args.surface_form)
                feature_path = direc + '/train_{}_features.pkl'.format(args.surface_form)
                if args.surface_form in ["attnmask"]:
                    example_path = direc + '/train_{}_examples.pkl'.format("digitctx")
                    feature_path = direc + '/train_{}_features.pkl'.format("digitctx")
            else:
                example_path = direc + '/train_{}_examples.pkl'.format(args.surface_form)
                feature_path = direc + '/train_{}_features.pkl'.format(args.surface_form)

            print("**** (train) Example_path ****: ", example_path)
            print("**** (train) Feature_path ****: ", feature_path)
            examples = read_file(example_path)
            drop_features = read_file(feature_path)
        else:
            if args.surface_form == "dice" or args.surface_form == "ctx_only":
                example_path = direc + '/eval_{}_examples.pkl'.format("local_ctx")
                feature_path = direc + '/eval_{}_features.pkl'.format("local_ctx")
            elif args.surface_form in ["diceloss"] and args.percent == "all":
                example_path = direc + '/eval_{}_examples.pkl'.format("dataset")
                feature_path = direc + '/eval_{}_features.pkl'.format("dataset")
            elif (args.surface_form in ["numeric", "textual"] or args.surface_form in ["dicemlp", "pretrain_ctx", "digitctx", "attnmask"]) and args.percent == "all":
                example_path = direc + '/eval_{}_examples.pkl'.format(args.surface_form)
                feature_path = direc + '/eval_{}_features.pkl'.format(args.surface_form)
                if args.surface_form in ["attnmask"]:
                    example_path = direc + '/eval_{}_examples.pkl'.format("digitctx")
                    feature_path = direc + '/eval_{}_features.pkl'.format("digitctx")
            else:
                if args.perturb_type in ["add10", "add100", "factor10", "factor100", "randadd100", "randfactor100"]:
                    example_path = direc + '/eval_{}_{}_examples.pkl'.format(args.surface_form, args.perturb_type)
                    feature_path = direc + '/eval_{}_{}_features.pkl'.format(args.surface_form, args.perturb_type)
                else:
                    example_path = direc + '/eval_{}_examples.pkl'.format(args.surface_form)
                    feature_path = direc + '/eval_{}_features.pkl'.format(args.surface_form)

            print("**** (eval) Example_path ****: ", example_path)
            print("**** (eval) Feature_path ****: ", feature_path)
            examples = read_file(example_path)
            drop_features = read_file(feature_path)
        
        self.max_dec_steps = len(drop_features[0].decoder_label_ids)
        
        features = []
        for i, (example, drop_feature) in tqdm(enumerate(zip(examples, drop_features)), desc="convert_to_input_features"):
            features.append(self.convert_to_input_features(example, drop_feature))
            if split == 'train' and args.num_train_samples >= 0 and len(features) >= args.num_train_samples:
                break

#         assert i == num_samples - 1
        self.num_samples = len(features)
        self.seq_len = drop_features[0].max_seq_length
        self.examples = examples
        self.drop_features = drop_features
        self.features = features
        self.example_ids = [f.example_id for f in features]
        self.input_ids = torch.tensor([f.input_ids for f in features]).long()
        self.input_mask = torch.tensor([f.input_mask for f in features]).long()
        self.segment_ids = torch.tensor([f.segment_ids for f in features]).long()
        self.label_ids = torch.tensor([f.label_ids for f in features]).long()
        self.head_type = torch.tensor([f.head_type for f in features]).long()
        self.q_spans = torch.tensor([f.q_spans for f in features]).long()
        self.p_spans = torch.tensor([f.p_spans for f in features]).long()
        # `number_in_input` and `number_indices` are for DiceEmbeddings to merge with [CTX] embeddings after encoding.
        self.numbers_in_input = torch.tensor([f.numbers_in_input for f in features]).float()  # Floating-point values exist
        self.number_indices = torch.tensor([f.number_indices for f in features]).long()
        self.ctx_indices = torch.tensor([f.ctx_indices for f in features]).long()
        self.ctx_labels = torch.tensor([f.ctx_labels for f in features]).long()
        self.digit_pos = torch.tensor([f.digit_pos for f in features]).long()
        self.digit_ent_indices = [f.digit_ent_indices for f in features]
        self.digit_type_indices = [f.digit_type_indices for f in features]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (self.input_ids[item], self.input_mask[item], self.segment_ids[item], self.label_ids[item],
                self.head_type[item], self.q_spans[item], self.p_spans[item], self.numbers_in_input[item], self.number_indices[item],
                self.ctx_indices[item], self.ctx_labels[item], self.digit_pos[item], self.digit_ent_indices[item], self.digit_type_indices[item])
    
    def convert_to_input_features(self, drop_example, drop_feature):
        # print(" ** convert_to_input_features ** ")

        max_seq_len = drop_feature.max_seq_length
        max_number_of_nums = max_seq_len // 2 if self.args.surface_form not in ["numeric", "textual"] else max_seq_len  # Maximum number of "numbers" in the given passage
        max_number_of_ctx = max_seq_len // 2  # Maximum number of "[CTX]" tokens in the given passage
        
        # input ids are padded by 0
        input_ids = drop_feature.input_ids
        input_ids += [IGNORE_IDX] * (max_seq_len - len(input_ids))
        
        # input mask is padded by 0
        input_mask = drop_feature.input_mask
        input_mask += [0] * (max_seq_len - len(input_mask))
        
        # segment ids are padded by 0
        segment_ids = drop_feature.segment_ids
        segment_ids += [0] * (max_seq_len - len(segment_ids))

        digit_pos = drop_feature.digit_pos
        
        # `digit_ent_indices` and `digit_type_indices` exists only when args.surface_form == 'localattn'
        digit_ent_indices = drop_feature.digit_ent_indices
        digit_type_indices = drop_feature.digit_type_indices

        # we assume dec label ids are already padded by 0s
        decoder_label_ids = drop_feature.decoder_label_ids
        assert len(decoder_label_ids) == self.max_dec_steps
        #decoder_label_ids += [0] * (MAX_DECODING_STEPS - len(decoder_label_ids))
        
        # for span extraction head, ignore idx == -1
        question_len = segment_ids.index(1) if 1 in segment_ids else len(segment_ids)
        starts, ends = drop_feature.start_indices, drop_feature.end_indices
        q_spans, p_spans = [], []
        for st, en in zip(starts, ends):
            if any([x < 0 or x >= max_seq_len for x in [st, en]]):
                continue
            elif all([x < question_len for x in [st, en]]):
                q_spans.append([st, en])
            elif all([question_len <= x for x in [st, en]]):
                p_spans.append([st, en])
        q_spans, p_spans = q_spans[:MAX_SPANS], p_spans[:MAX_SPANS]
        head_type = 1 if q_spans or p_spans else -1
        q_spans += [[-1,-1]]*(MAX_SPANS - len(q_spans))
        p_spans += [[-1,-1]]*(MAX_SPANS - len(p_spans))

        numbers_in_input = []
        number_indices = []
        numbers_in_input = drop_feature.numbers_in_input
        number_indices = drop_feature.number_indices
        num_pad = -1

        if len(numbers_in_input) < max_number_of_nums:
            pad_length_numbers = max_number_of_nums - len(numbers_in_input)
            numbers_in_input += [num_pad] * pad_length_numbers
        else:
            numbers_in_input = numbers_in_input[:max_number_of_nums]

        if len(number_indices) < max_number_of_nums and len(digit_pos) < max_number_of_nums:
            pad_length_indices = max_number_of_nums - len(number_indices)
            number_indices += [num_pad] * pad_length_indices
            digit_pos += [num_pad] * pad_length_indices
        else:
            number_indices = number_indices[:max_number_of_nums]
            digit_pos = digit_pos[:max_number_of_nums]

        # print("numbers_in_input: ", len(numbers_in_input))
        # print("numbers_indices: ", len(number_indices))
        # print("digit_pos: ", len(digit_pos))
        
        try:
            assert len(numbers_in_input) == len(number_indices) == len(digit_pos)
        except AssertionError as e:
            print("numbers_in_input: ", len(numbers_in_input))
            print("numbers_indices: ", len(number_indices))
            print("digit_pos: ", len(digit_pos))
            print(e)
        
        ctx_indices = []
        ctx_labels = []
        if self.args.surface_form in ["pretrain_ctx", "digitctx", "attnmask"]:
            ctx_indices = drop_feature.ctx_indices
            ctx_labels = drop_feature.ctx_labels
            if len(ctx_indices) < max_number_of_ctx:
                ctx_pad = -1
                ctx_indices += [ctx_pad] * (max_number_of_ctx - len(ctx_indices))
                ctx_labels += [ctx_pad] * (max_number_of_ctx - len(ctx_labels))

        return ModelFeatures(drop_feature.example_index, input_ids, input_mask, segment_ids,
                             decoder_label_ids, head_type, q_spans, p_spans, numbers_in_input, number_indices, ctx_indices, ctx_labels, digit_pos, digit_ent_indices, digit_type_indices)


def collate_batch(batch):
    '''
    batch - List of DropFeature instances of length batch_size
    '''
    input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans = [], [], [], [], [], [], []
    numbers_in_input, number_indices, ctx_indices, ctx_labels, digit_pos, digit_ent_indices, digit_type_indices = [], [], [], [], [], [], []
    for i in range(len(batch)):
        input_ids.append(batch[i][0])
        input_mask.append(batch[i][1])
        segment_ids.append(batch[i][2])
        label_ids.append(batch[i][3])
        head_type.append(batch[i][4])
        q_spans.append(batch[i][5])
        p_spans.append(batch[i][6])
        numbers_in_input.append(batch[i][7])
        number_indices.append(batch[i][8])
        ctx_indices.append(batch[i][9])
        ctx_labels.append(batch[i][10])
        digit_pos.append(batch[i][11])
        digit_ent_indices.append(batch[i][12])
        digit_type_indices.append(batch[i][13])

    input_ids = torch.stack(input_ids)
    input_mask = torch.stack(input_mask)
    segment_ids = torch.stack(segment_ids)
    label_ids = torch.stack(label_ids)
    head_type = torch.stack(head_type)
    q_spans = torch.stack(q_spans)
    p_spans = torch.stack(p_spans)
    numbers_in_input = torch.stack(numbers_in_input)
    number_indices = torch.stack(number_indices)
    ctx_indices = torch.stack(ctx_indices)
    ctx_labels = torch.stack(ctx_labels)

    new_batch = (input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,
                 numbers_in_input, number_indices, ctx_indices, ctx_labels, digit_pos, digit_ent_indices, digit_type_indices)
    
    return new_batch

def make_output_dir(args, scripts_to_save=[sys.argv[0]]):
    # scripts_to_save are relative paths of files to save
    os.makedirs(args.output_dir, exist_ok=True)
    tb_dir = os.path.join(args.output_dir, 'log')
    # remove prev tensorboard logs
    shutil.rmtree(tb_dir, ignore_errors=True)
    code_dir = os.path.join(args.output_dir, 'scripts')
    os.makedirs(code_dir, exist_ok=True)
    for script in scripts_to_save:
        dst_file = os.path.join(code_dir, os.path.basename(script))
        shutil.copyfile(script, dst_file)


def replace_to_dice(args, tokenizer: BertTokenizer, embeddings: torch.nn.Embedding, model):
    # Turn 0~9 into DiceEmbeddings (in torch.tensor) (min_bound=0, max_bound=9)
    dice = DiceEmbedding(model.config.hidden_size, min_bound=args.min_bound, max_bound=args.max_bound)
    # Identify the indices of the 0~9 embeddings in `embeddings`
    digits = [str(d) for d in list(range(10))]
    idx2digit = {}
    digit2idx = {}
    for token, idx in tokenizer.get_vocab().items():
        if token in digits:
            idx2digit[idx] = token
            digit2idx[token.strip()] = idx
    # Replace the embeddings
    # Here, we do not deal with ##0, ##1, ..., ##9 (because they are redundant with 0-9 embeddings)
    dice_embeddings = {}
    if os.path.exists(os.path.join(args.dice_dir, args.dice_path)):
        dice_embeddings = torch.load(os.path.join(args.dice_dir, args.dice_path))
        logger.info("digit-DICE embedding loaded from {}".format(os.path.join(args.dice_dir, args.dice_path)))
    else:
        if not os.path.isdir(args.dice_dir):
            os.mkdir(args.dice_dir)
        for idx, digit in idx2digit.items():
            dice_embeddings[digit] = dice.make_dice(int(digit))
        torch.save(dice_embeddings, os.path.join(args.dice_dir, args.dice_path))
        logger.info("digit-DICE embedding saved at {}".format(os.path.join(args.dice_dir, args.dice_path)))
    
    digit_list, dice_list = map(list, zip(*dice_embeddings.items()))
    digit_idx_list = [digit2idx[d] for d in digit_list]

    embeddings.weight.data[digit_idx_list] = torch.FloatTensor(dice_list)

    logger.info("** COMPLETE : Digit embeddings (0~9) replaced with DICE. **")

    return idx2digit, embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--examples_n_features_dir",
                        default='data/examples_n_features/',
                        type=str,
                        help="Dir containing drop examples and features.")
    parser.add_argument("--mlm_dir",
                        default='../data/MLM_train/',
                        type=Path,
                        help="The data dir with MLM taks. Should contain the .jsonl files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default='./out_drop_finetune',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model", default="bert-base-uncased", type=str)
    parser.add_argument("--init_weights_dir",
                        default='',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")
    parser.add_argument("--max_seq_length",
                        default=-1,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_inference",
                        action='store_true',
                        help="Whether to run inference on the dev set.")
    parser.add_argument("--do_pretrain_ctx",
                        action='store_true',
                        help="Whether to pretrain [CTX] with binary classification : number vs. non-number")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlm_batch_size",
                        default=-1,
                        type=int,
                        help="Total batch size for mlm train data.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_samples",
                        default=-1,
                        type=int,
                        help="Total number of training samples used.")
    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--freeze_encoder',
                        action='store_true',
                        help="Whether to freeze the bert encoder, embeddings.")
    parser.add_argument('--indiv_digits',
                        action='store_true',
                        help="Whether to tokenize numbers as digits.")
    parser.add_argument('--rand_init',
                        action='store_true',
                        help="Whether to use random init instead of BERT.")
    parser.add_argument('--random_shift',
                        action='store_true',
                        help="Whether to randomly shift position ids of encoder input.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--mlm_scale',
                        type=float, default=1.0,
                        help="mlm loss scaling factor.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--surface_form', required=True, type=str, help="10ebased, 10based, character, edigit, ...")
    parser.add_argument("--percent", required=True, type=str, help="Percentage of drop_dataset_train for sample efficiency test.")
    parser.add_argument("--perturb_type", type=str, default="_", help="Amount of perturbation (e.g., add10, factor10, add100, factor100, etc.)")
    parser.add_argument("--log_eval_step", type=int, default=-1, help="The number of steps after train iteration for evaluation.")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden_size for DiceEmbedding.")
    parser.add_argument("--dice_dir", type=str, default="dice_embeddings", help="Directory path for DiceEmbeddings pre-computed for DROP")
    parser.add_argument("--dice_path", type=str, default="dice_embeddings_drop.pth", help="File path for DiceEmbeddings pre-computed for DROP")
    parser.add_argument("--min_bound", type=int, default=0, help="DiceEmbedding minimum bound")
    parser.add_argument("--max_bound", type=int, default=9999, help="DiceEmbedding maximum bound")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        if args.do_pretrain_ctx:
            pass
        else:
            raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        make_output_dir(args, scripts_to_save=[sys.argv[0], 'modeling.py', 'create_examples_n_features.py'])

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer_dir = "bert_tokenizer"
    if args.do_eval and args.do_inference:
        tokenizer_path = os.path.join(args.init_weights_dir, tokenizer_dir)
        if os.path.isdir(tokenizer_path) and len(os.listdir(tokenizer_path)) > 0:
            logger.info("BertTokenizer loaded from [ {} ]".format(tokenizer_path))
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    elif os.path.isdir(os.path.join(args.output_dir, tokenizer_dir)) and len(os.listdir(os.path.join(args.output_dir, tokenizer_dir))) > 0:
        logger.info("BertTokenizer loaded from [ {} ]".format(os.path.join(args.output_dir, tokenizer_dir)))
        tokenizer = BertTokenizer.from_pretrained(os.path.join(args.output_dir, tokenizer_dir), do_lower_case=True)
    else:
        logger.info("BertTokenizer loaded from [ {} ]".format("bert-base-uncased"))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if not os.path.isdir(os.path.join(args.output_dir, tokenizer_dir)):
        os.mkdir(os.path.join(args.output_dir, tokenizer_dir))

    if len(os.listdir(os.path.join(args.output_dir, tokenizer_dir))) == 0 and tokenizer.add_tokens([CONTEXT_TOKEN]):  # "[CTX]" at index 30522
        logger.info("{} added to tokenizer - config.vocab_size : {}".format(CONTEXT_TOKEN, len(tokenizer)))
        tokenizer.save_pretrained(os.path.join(args.output_dir, tokenizer_dir))
        logger.info("Tokenizer saved at {}".format(os.path.join(args.output_dir, tokenizer_dir)))

    if args.init_weights_dir:
        if args.do_pretrain_ctx:
            model = BertCtxClassifier.from_pretrained(args.init_weights_dir)  # Model for pre-training [CTX] w/ binary classification
        else:
            model = BertTransformer.from_pretrained(args.init_weights_dir)
    else:
        # prepare model
        model = BertTransformer.from_pretrained(args.model,
            cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))
    
    resized_embeddings = model.resize_token_embeddings(len(tokenizer))
    logger.info("Embedding number: %d", resized_embeddings.num_embeddings)

    # assert resized_embeddings.num_embeddings == model.embed.num_embeddings == model.dec_head.decoder.out_features == model.cls.predictions.decoder.out_features

    if args.surface_form in ["skipconnect", "skipctx", "dicemlp", "diceembed", "diceembed9999", "digitctx", "numeric", "textual"]:
        # Replace digit embeddings (0~9) to DiceEmbeddings
        idx2digit, embeddings_pointer = replace_to_dice(args, tokenizer, resized_embeddings, model)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    eval_data = DropDataset(args, 'eval')
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_batch)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    if args.do_eval and args.do_inference:
        inference(args, model, eval_dataloader, device, tokenizer)
        
    if args.do_train:
        # Prepare data loader
        train_data = DropDataset(args, 'train')
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_batch)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that breaks apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     warmup=args.warmup_proportion,
                     t_total=num_train_optimization_steps)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        '''
        ------------------------------------------------------------------------------
        TODO: check training resume, fp16, use --random_shift for short inputs
        ------------------------------------------------------------------------------
        '''
        # using fp16
        fp16 = False
#         try:
#             from apex import amp
#             model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#             fp16 = True
#         except ImportError:
#             logger.info("Not using 16-bit training due to apex import error.")
        
#         if n_gpu > 1:
#             model = torch.nn.DataParallel(model)
        
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'log'))  # tensorboard
        
        # masked LM data
        do_mlm_task = False
        if args.mlm_batch_size > 0:
            if not os.path.isdir(args.mlm_dir):
                os.mkdir(args.mlm_dir)
            mlm_dataset = MLMDataset(training_path=args.mlm_dir, tokenizer=tokenizer)
            mlm_dataloader = DataLoader(mlm_dataset, sampler=RandomSampler(mlm_dataset), batch_size=args.mlm_batch_size)
            mlm_iter = iter(mlm_dataloader)
            do_mlm_task = True
        
        model.train()
        (global_step, all_losses, all_errors, all_dec_losses, all_dec_errors, eval_errors,
         best, best_mlm, t_prev, do_eval) = 0, [], [], [], [], [], 1000, 1000, time.time(), False
        mlm_losses, mlm_errors, all_span_losses, all_span_errors, all_dice_losses, all_scalar_losses = [], [], [], [], [], []
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # grads wrt to train data
                digit_pos = batch[-3]
                digit_ent_indices = batch[-2]
                digit_type_indices = batch[-1]
                batch = tuple(t.to(device) for t in batch[:-3])
                input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans, numbers_in_input, number_indices, ctx_indices, ctx_labels = batch

                # print("text: ", tokenizer.convert_ids_to_tokens(input_ids[0]))

                losses = model(input_ids, segment_ids, input_mask, numbers_in_input, number_indices, ctx_indices, digit_pos, digit_ent_indices, digit_type_indices,
                               random_shift=args.random_shift, target_ids=label_ids, target_mask=None, answer_as_question_spans=q_spans, answer_as_passage_spans=p_spans,
                               head_type=head_type, dice_mode=args.surface_form, dice_load=True, tokenizer=tokenizer)
                loss, errs, dec_loss, dec_errors, span_loss, span_errors, type_loss, type_errors, type_preds, dice_loss, scalar_loss = losses
                
                # aggregate on multi-gpu
                take_mean = lambda x: x.mean() if x is not None and sum(x.size()) > 1 else x
                take_sum = lambda x: x.sum() if x is not None and sum(x.size()) > 1 else x
                [loss, dec_loss, span_loss, type_loss, dice_loss, scalar_loss] = list(map(take_mean, [loss, dec_loss, span_loss, type_loss, dice_loss, scalar_loss]))
                [errs, dec_errors, span_errors, type_errors] = list(map(take_sum, [errs, dec_errors, span_errors, type_errors]))

                # Add the dice auxiliary loss term + scalar auxiliary loss term
                if dice_loss is not None:
                    loss += dice_loss
                    all_dice_losses.append(dice_loss.item())
                if scalar_loss is not None:
                    loss += scalar_loss
                    all_scalar_losses.append(scalar_loss.item())
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                all_losses.append(loss.item()); all_dec_losses.append(dec_loss.item()); 
                all_errors.append(errs.item() / input_ids.size(0))
                all_dec_errors.append(dec_errors.item() / input_ids.size(0))
                all_span_losses.append(span_loss.item()); all_span_errors.append(span_errors.item() / input_ids.size(0))
         
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if do_mlm_task:
                    # grads wrt to mlm data
                    while True:
                        try:
                            batch = next(mlm_iter)  # sample next mlm batch
                            break
                        except StopIteration:       # end of epoch: reset and shuffle
                            mlm_iter = iter(mlm_dataloader)
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, label_ids = batch
                    loss, errs = model(input_ids, None, input_mask, target_ids=label_ids, 
                                       ignore_idx=IGNORE_IDX, task='mlm')
                    loss, err_sum = take_mean(loss), take_sum(errs)      # for multi-gpu
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss = args.mlm_scale * loss
                    mlm_losses.append(loss.item()); mlm_errors.append(err_sum.item() / input_ids.size(0))

                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                else:
                    mlm_losses.append(-1); mlm_errors.append(-1)
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Before updating, zero out the gradients for digit-level DiceEmbeddings (0~9) in nn.Embedding (of the model)
                    if args.surface_form in ["skipconnect", "skipctx", "dicemlp", "diceembed", "diceembed9999", "digitctx", "numeric", "textual"]:  # TODO: May need to comment out upon need
                        embeddings_pointer.weight.grad[list(idx2digit.keys())] = 0.0
                    optimizer.step()          # update step
                    optimizer.zero_grad()
                    global_step += 1
                    
                train_result = {'trn_loss': all_losses[-1], 'trn_dec_loss': all_dec_losses[-1], 
                                'trn_err': all_errors[-1], 'trn_dec_err': all_dec_errors[-1], 
                                'lr': optimizer.get_lr()[0], 'trn_span_loss': all_span_losses[-1], 
                                'trn_span_err': all_span_errors[-1], 'epoch': epoch}
                mlm_result = {'trn_mlm_loss': mlm_losses[-1], 'trn_mlm_err': mlm_errors[-1]}
                tb_writer.add_scalars('train', train_result, len(all_losses))
                tb_writer.add_scalars('mlm', mlm_result, len(all_losses)) if do_mlm_task else None

                # if time.time() - t_prev > 60*60: # evaluate every hr
                #     do_eval = True
                #     t_prev = time.time()

                start_eval = do_eval if args.log_eval_step == -1 else (step + 1) % args.log_eval_step == 0
                # start_eval = True
                if start_eval:
                    do_eval = False
                    eval_result = evaluate(args, model, eval_dataloader, device, len(train_data), tokenizer)
                    eval_err = eval_result['eval_err']
                    if eval_err < best or (eval_err < best + 0.005 and np.mean(mlm_errors[-1000:]) < best_mlm):
                        # if eval err is in range of best, look at MLM err
                        train_state = {'global_step': global_step, 'optimizer_state_dict': optimizer.state_dict()}
                        train_state.update(train_result)
                        save(args, model, tokenizer, train_state)
                        best_mlm = min(best_mlm, np.mean(mlm_errors[-1000:]))
                    best = min(best, eval_err)
                    eval_errors.append((len(all_losses), eval_err))
                    model.train()
            
                    tb_writer.add_scalars('eval', eval_result, len(all_losses))
#                     for name, param in model.named_parameters():
#                         tb_writer.add_histogram(name, param.clone().cpu().data.numpy(), len(all_losses))
            # end of epoch
            do_eval = True

        # training complete
        tb_writer.export_scalars_to_json(os.path.join(args.output_dir, 'training_scalars.json'))
        tb_writer.close()

    if args.do_pretrain_ctx:
        '''
        Pre-training step to utilize [CTX] - binary classification to classify `number` and `non-number` for every [CTX] token replacing the original tokens
        '''
        # Prepare data loader
        train_data = DropDataset(args, 'train')
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that breaks apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     warmup=args.warmup_proportion,
                     t_total=num_train_optimization_steps)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        # using fp16
        fp16 = False

        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'log'))  # tensorboard

        model.train()
        (global_step, all_losses, eval_errors, best, best_mlm, t_prev, do_eval) = 0, [], [], 1000, 1000, time.time(), False
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # grads wrt to train data
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans, numbers_in_input, number_indices, ctx_indices, ctx_labels = batch

                outputs = model(input_ids, segment_ids, input_mask, random_shift=args.random_shift, ignore_idx=-1, labels=ctx_labels, ctx_indices=ctx_indices)
                loss, logits = outputs
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                all_losses.append(loss.item())

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()          # update step
                    optimizer.zero_grad()
                    global_step += 1
                    
                train_result = {'trn_loss': all_losses[-1], 'lr': optimizer.get_lr()[0], 'epoch': epoch}

                tb_writer.add_scalars('train', train_result, len(all_losses))

                # if do_eval or (step + 1) % args.log_eval_step == 0:
                if do_eval:
                    do_eval = False
                    eval_result = evaluate_pretrain_ctx(args, model, eval_dataloader, device, len(train_data))
                    eval_err = eval_result['eval_err']
                    if eval_err < best or (eval_err < best + 0.005 and np.mean(mlm_errors[-1000:]) < best_mlm):
                        # if eval err is in range of best, look at MLM err
                        train_state = {'global_step': global_step, 'optimizer_state_dict': optimizer.state_dict()}
                        train_state.update(train_result)
                        save(args, model, tokenizer, train_state)
                    best = min(best, eval_err)
                    eval_errors.append((len(all_losses), eval_err))
                    model.train()
            
                    tb_writer.add_scalars('eval', eval_result, len(all_losses))
#                     for name, param in model.named_parameters():
#                         tb_writer.add_histogram(name, param.clone().cpu().data.numpy(), len(all_losses))
            # end of epoch
            do_eval = True

        # training complete
        tb_writer.export_scalars_to_json(os.path.join(args.output_dir, 'training_scalars.json'))
        tb_writer.close()


def evaluate_pretrain_ctx(args, model, eval_dataloader, device, n_train):
    # Evaluation code for the pre-training of [CTX] special token with the binary classification task (number vs. non-number)
    model.eval()
    eval_examples = eval_dataloader.dataset.examples
    predictions, eval_loss, eval_accuracy, eval_err_sum = {}, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    sample_accs = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans, numbers_in_input, number_indices, ctx_indices, ctx_labels = batch
        
        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask, random_shift=args.random_shift, ignore_idx=-1, labels=ctx_labels, ctx_indices=ctx_indices)
            loss, logits = outputs
            tmp_eval_loss = loss

        def calculate_ctx_accuracy(logits, ctx_labels, ctx_indices):
            tmp_acc = 0.0
            for i, batch in enumerate(logits):
                ctx_labels_np = ctx_labels[i].cpu().data.numpy()
                pad_start_idx = np.where(ctx_labels_np == -1)[0][0]  # First index of pad_idx in `ctx_indices`
                pred = torch.argmax(batch[:pad_start_idx], dim=-1)
                tmp_acc += (pred == ctx_labels[i][:pad_start_idx]).float().mean().item()
            return tmp_acc
        
        tmp_eval_accuracy = calculate_ctx_accuracy(logits, ctx_labels, ctx_indices)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples  # TODO: eval_accuracy and DropEM difference is too big.
    eval_err_sum /= nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_acc': eval_accuracy,
              'eval_err': 1 - eval_accuracy}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\t" % (key, str(result[key])))
        writer.write("n_train = %d\t" % n_train)
        writer.write("\n")

    write_file(sample_accs, os.path.join(args.output_dir, "ems.jsonl"))

    return result


def save(args, model, tokenizer, train_state_dict):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    output_args_file = os.path.join(args.output_dir, 'training_args.bin')
    train_state_file = os.path.join(args.output_dir, 'training_state.bin')
    
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)
    torch.save(args, output_args_file)
#     torch.save(train_state_dict, train_state_file)


def evaluate(args, model, eval_dataloader, device, n_train, tokenizer):
    model.eval()
    eval_examples = eval_dataloader.dataset.examples
    predictions, eval_loss, eval_accuracy, eval_err_sum = {}, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    sample_accs = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        digit_pos = batch[-3]
        digit_ent_indices = batch[-2]
        digit_type_indices = batch[-1]
        batch = tuple(t.to(device) for t in batch[:-3])
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans, numbers_in_input, number_indices, ctx_indices, ctx_labels = batch
        
        with torch.no_grad():
            losses = model(input_ids, segment_ids, input_mask, numbers_in_input, number_indices, ctx_indices, digit_pos, digit_ent_indices, digit_type_indices, random_shift=args.random_shift, target_ids=label_ids,
                           target_mask=None, answer_as_question_spans=q_spans, answer_as_passage_spans=p_spans,
                           head_type=head_type, dice_mode=args.surface_form, dice_load=True, tokenizer=tokenizer)
            loss, errs, dec_loss, dec_errors, span_loss, span_errors, type_loss, type_errors, type_preds, dice_loss, scalar_loss = losses

            tmp_eval_loss = loss
            if dice_loss is not None:
                tmp_eval_loss += dice_loss
            if scalar_loss is not None:
                tmp_eval_loss += scalar_loss

        for i, sample_acc in enumerate((errs == 0).cpu().tolist()):
            sample_accs.append({'qid': eval_examples[i+nb_eval_examples].qas_id, 'em': sample_acc})
        
        tmp_eval_accuracy = (errs == 0).sum().item()

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        eval_err_sum += dec_errors.sum().item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples  # TODO: eval_accuracy and DropEM difference is too big.
    eval_err_sum /= nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_acc': eval_accuracy,
              'eval_dec_err_sum': eval_err_sum,
              'eval_err': 1-eval_accuracy}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\t" % (key, str(result[key])))
        writer.write("n_train = %d\t" % n_train)
        writer.write("\n")

    write_file(sample_accs, os.path.join(args.output_dir, "ems.jsonl"))

    return result


def inference(args, model, eval_dataloader, device, tokenizer):

    logger.info("inference() start")

    model.eval()
    eval_examples = eval_dataloader.dataset.examples
    eval_drop_features = eval_dataloader.dataset.drop_features
    [start_tok_id, end_tok_id] = tokenizer.convert_tokens_to_ids([START_TOK, END_TOK]) # [1030, 1032]
    all_dec_ids, all_label_ids, all_type_preds, all_start_preds, all_end_preds, all_type_logits = [], [], [], [], [], []
    all_input_ids = []
    nb_eval_examples, eval_accuracy, eval_err_sum = 0, 0, 0
    
    for batch in tqdm(eval_dataloader, desc="Inference"):
        digit_pos = batch[-3]
        digit_ent_indices = batch[-2]
        digit_type_indices = batch[-1]
        batch = tuple(t.to(device) for t in batch[:-3])
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans, numbers_in_input, number_indices, ctx_indices, ctx_labels = batch

        with torch.no_grad():
            out = model(input_ids, segment_ids, input_mask=input_mask, numbers_in_input=numbers_in_input, number_indices=number_indices,
                        digit_ent_indices=digit_ent_indices, digit_type_indices=digit_type_indices, random_shift=args.random_shift,
                        task='inference', dice_mode=args.surface_form, dice_load=True, max_decoding_steps=eval_dataloader.dataset.max_dec_steps)
            # here segment_ids are only used to get the best span prediction
            dec_preds, type_preds, start_preds, end_preds, type_logits = tuple(t.cpu() for t in out)
            # dec_preds: [bsz, max_deocoding_steps], has start_tok
#           # type_preds, start_preds, end_preds, type_logits : [bsz], [bsz], [bsz], [bsz, 2]
        assert dec_preds.size() == label_ids.size()
        assert dec_preds.dim() == 2
        
#         bch_errs = ((bch_preds != target_ids).float() * (target_ids != IGNORE_IDX).float()).sum(dim=-1)
#         bch_eval_accuracy = (bch_errs == 0).sum().item()
#         eval_accuracy += bch_eval_accuracy
#         eval_err_sum += bch_errs.sum().item() # all errors
        nb_eval_examples += input_ids.size(0)
        all_dec_ids.append(dec_preds); all_label_ids.append(label_ids); all_type_preds.append(type_preds)
        all_start_preds.append(start_preds); all_end_preds.append(end_preds); all_type_logits.append(type_logits)
        all_input_ids.append(input_ids)
        #break
#     eval_accuracy /= nb_eval_examples
#     eval_err_sum /= nb_eval_examples
#     result = {'eval_accuracy': eval_accuracy,
#               'eval_err_sum': eval_err_sum}

#     logger.info("***** Eval results *****")
#     for key in sorted(result.keys()):
#         logger.info("  %s = %s", key, str(result[key]))
    
    tup = all_dec_ids, all_label_ids, all_type_preds, all_start_preds, all_end_preds, all_type_logits, all_input_ids
    all_dec_ids, all_label_ids, all_type_preds, all_start_preds, all_end_preds, all_type_logits, all_input_ids = \
                                                                tuple(torch.cat(t, dim=0).tolist() for t in tup)
    
    def trim(ids):
        # remove start tok
        ids = ids[1:] if ids[0] == start_tok_id else ids
        # only keep predictions until the first pad/end token
        _ids = []
        for id in ids:
            if id in [IGNORE_IDX, end_tok_id]:
                break
            else:
                _ids.append(id)
        return _ids
    def process(text):
        processed = '.'.join([x.strip() for x in text.split('.')]) # remove space around decimal
        try:
            float(processed)  #'.' is a decimal only if final str is a number
        except ValueError:
            processed = text
        return '-'.join([x.strip() for x in processed.split('-')]) # remove space around "-"
    def revive_numbers():
        pass
    
    predictions, ems, drop_ems, drop_f1s = [], [], [], []
    for i in range(len(all_dec_ids)):
        example = eval_examples[i]
        drop_feature = eval_drop_features[i]
        answer_text = (SPAN_SEP+' ').join(example.answer_texts).strip().lower()
        
        # # TODO: If answer_type == "number", reverse the dec_id sequence
        # if example.answer_type == "number":
        #     answer_texts = []
        #     if isinstance(example.answer_texts, list):
        #         for ans_txt in example.answer_texts:
        #             answer_texts.append(ans_txt[::-1])  # Reverse the number answer
        #     answer_text = (SPAN_SEP+' ').join(answer_texts).strip()
        #     # print("example.answer_text: {} / answer_text: {}".format(example.answer_texts, answer_text))

        processed_answer_text = process(answer_text)
        # generator prediction
        dec_ids = trim(all_dec_ids[i])
        dec_toks = tokenizer.convert_ids_to_tokens(dec_ids)
        dec_text = detokenize(dec_toks)
        dec_processed = process(dec_text)
        # span prediction
        start_pred, end_pred, input_ids = all_start_preds[i], all_end_preds[i], all_input_ids[i]
        [start_pred, end_pred] = sorted([start_pred, end_pred])
        span_ids = [x for x in input_ids[start_pred:end_pred+1] if x != 0]
        span_toks = tokenizer.convert_ids_to_tokens(span_ids)
        span_text = detokenize(span_toks)
        span_processed = process(span_text)
        
        span_pred, used_orig = wrapped_get_final_text(example, drop_feature, start_pred, end_pred)
        if not used_orig:
            span_pred = process(span_pred)
    
        prediction = span_pred if all_type_preds[i] else dec_processed
        head_pred = 'span_extraction' if all_type_preds[i] else 'generator'

        # if i == 10:
        #     exit()
        
        # compute drop em and f1
        drp = DropEmAndF1()

        drp(prediction, example.answer_annotations)
        drop_em, drop_f1 = drp.get_metric()
        em = exact_match_score(prediction, processed_answer_text)

        # Deal with -yard and number matches (e.g., 40-yard == 40)
        # Any decoder-generated answers containing the `processed_answer_text` should be considered an answer.
        if dec_processed == processed_answer_text or dec_processed[:len(processed_answer_text)] == processed_answer_text or \
           span_pred == processed_answer_text or span_text == processed_answer_text or processed_answer_text[:len(dec_processed)] == dec_processed:
            drop_em = 1.0
            drop_f1 = 1.0
            em = 1.0
        
        predictions.append({'query_id': example.qas_id, 'passage_id':example.passage_id, 
                            'processed_dec_out': dec_processed, 'prediction': prediction, 
                            'ans_used': processed_answer_text, 'type_logits': all_type_logits[i],
                            'head_pred': head_pred, 'processed_span_out': span_processed,
                            'dec_out': dec_text, 'span_out': span_text, 'span_pred': span_pred,
                            'drop_em': drop_em, 'drop_f1': drop_f1, 'em': em})
        ems.append(em); drop_ems.append(drop_em)
        drop_f1s.append(drop_f1)
        if i < 10:
            print(prediction, processed_answer_text, end=' || ')
    logger.info(f'EM: {np.mean(ems)}, Drop EM: {np.mean(drop_ems)}')
    logger.info(f'F1: {np.mean(drop_f1s)}')
    logger.info('saving predictions.jsonl in ' + args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    em_f1 = {}
    em_f1["em"] = np.mean(ems)
    em_f1["drop_em"] = np.mean(drop_ems)
    em_f1["f1"] = np.mean(drop_f1s)
    
    write_file(predictions, os.path.join(args.output_dir, "{}_{}_predictions.jsonl".format(args.surface_form, args.percent)))
    write_file(em_f1, os.path.join(args.output_dir, "{}_{}_em_f1.json".format(args.surface_form, args.percent)))
    
def detokenize(tok_tokens):
    tok_tokens = [tok for tok in tok_tokens if tok.strip().lower() != CONTEXT_TOKEN.lower()]  # Remove [CTX] token
    
    def append_digits2numbers(tok_tokens):
        digits = [str(d) for d in list(range(10))]
        digits += ["##" + d for d in digits]
        is_number = False
        new_tok_tokens = []
        for i, tok in enumerate(tok_tokens):
            # Need to take into account the "." characters for "floating-point numbers"
            if tok.strip() in digits or (tok.strip() == "." and i+1 < len(tok_tokens) and tok_tokens[i-1].isdigit() and tok_tokens[i+1].isdigit()):
                tok = tok.strip().replace(" ##", "")  # ##3, 2, ##4
                tok = tok.strip().replace("##", "")
                if is_number:
                    new_tok_tokens[-1] += tok
                else:
                    new_tok_tokens.append(tok)
                    is_number = True
            else:
                new_tok_tokens.append(tok)
                is_number = False
        return new_tok_tokens

    # Aggregate digits to numbers
    tok_tokens = append_digits2numbers(tok_tokens)
    if all(tok.isdigit() for tok in tok_tokens) or re.findall("\d+\.\d+", "".join(tok_tokens)):
        tok_text = "".join(tok_tokens)

    tok_text = " ".join(tok_tokens)
    
    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text


def detokenize_original(tok_tokens):
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text


def wrapped_get_final_text(example, feature, start_index, end_index, do_lower_case=True, verbose_logging=False, logger=None):
    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    tok_tokens = [tok for tok in tok_tokens if tok not in ['[PAD]', '[CLS]', '[SEP]']]
    tok_text = detokenize(tok_tokens)
    
    if start_index in feature.doc_token_to_orig_map and end_index in feature.doc_token_to_orig_map:
        orig_doc_start = feature.doc_token_to_orig_map[start_index]
        orig_doc_end = feature.doc_token_to_orig_map[end_index]
        orig_tokens = example.orig_passage_tokens[orig_doc_start:(orig_doc_end + 1)]
    elif start_index in feature.que_token_to_orig_map and end_index in feature.que_token_to_orig_map:
        orig_que_start = feature.que_token_to_orig_map[start_index]
        orig_que_end = feature.que_token_to_orig_map[end_index]
        orig_tokens = example.orig_question_tokens[orig_que_start:(orig_que_end + 1)]
    else:
        orig_tokens = None
        return tok_text, False
    
    orig_text = " ".join(orig_tokens)
    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, logger)
    return final_text, True


if __name__ == "__main__":
    main()



'''
(for pre-training on short texts use --random_shift, for finetuning use mlm_batch_size -1)

CUDA_VISIBLE_DEVICES=4,5,6,7 python finetune_on_drop.py --do_eval --do_inference --examples_n_features_dir ./data/examples_n_features/ --eval_batch_size 1000 --init_weights_dir out_drop_finetune_bert  --output_dir preds
'''

# export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./log

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import collections, jsonlines
import re

from random import random, randrange, randint, shuffle, choice, sample
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import RobertaTokenizer, AlbertTokenizer, BertTokenizer
import numpy as np
import ujson as json


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def is_start_piece(wp):
    if wp.startswith('▁'):
        return True
    else:
        return False


def create_masked_lm_predictions_bert(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
#         # If adding a whole-word mask would exceed the maximum number of
#         # predictions, then just skip this candidate.
#         if len(masked_lms) + len(index_set) > num_to_mask:
#             continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
            
        op = np.random.choice(['mask', 'orig', 'random'], p=[0.8, 0.1, 0.1])
        for index in index_set:
            covered_indexes.add(index)

            # 80% of the time, replace with [MASK]
            if op == 'mask':
                masked_token = "[MASK]"
            elif op == 'orig':
                # 10% of the time, keep original
                masked_token = tokens[index]
            else:
                # 10% of the time, replace with random word
                masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

#     assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def create_masked_lm_predictions_albert(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective."""
    do_whole_word_mask = True
    ngram = True
    favor_shorter_ngram = True
    do_permutation = True

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and not is_start_piece(token)):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(token):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    if not favor_shorter_ngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx : idx + n])
        ngram_indexes.append(ngram_index)

    shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(ngrams[:len(cand_index_set)],
                            p=pvals[:len(cand_index_set)] /
                            pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_list[randint(0, len(vocab_list) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)


def create_instances_from_document(
        model_prefix, all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):

    random_next_sentence = True  # FLAG for choosing random sentences from another document (or just swap with the subsequent sentence) for sop
    document = all_documents[document_index]
    # document is a list of toknized sents
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is the hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []  # `current_chunk` is a list of sentences
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        if len(segment) >= 1:
            # '■' is a delimiter between paragraphs
            # (ref: https://github.com/ag1988/injecting_numeracy/tree/master/pre_training)
            if '■' in segment:
                segment = ["■"]
            current_chunk.append(segment)  # append to sents to current_chunk until target_seq_length
        current_length += len(segment)

        # print("current doc: ", document)
        if random() < 0.5:
            random_next_sentence = True
        else:
            random_next_sentence = False

        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` should be assigned to `A` (the first sentence)
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
            
                tokens_b = []
                # `is_random_next` determines the label for sentence order prediction (sop)
                case = 0
                is_random_next = False
                if len(current_chunk) == 1 or (random_next_sentence and random() < 0.5):
                    case = 1
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # If the corpus size is large enough, the iteration will not go for more than 10 iters
                    if len(all_documents) > 1:
                        for _ in range(10):
                            random_document_index = randint(0, len(all_documents) - 1)
                            if random_document_index != document_index and len(all_documents[random_document_index]) > 1:
                                break
                    
                    random_document = all_documents[random_document_index]
                    random_start = randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        if len(random_document[j]) >= 1:
                            tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    
                    # Make sure that `tokens_b` is not empty from selecting empty chunks in `random_document`
                    if len(tokens_b) < 1:
                        for _ in range(10):
                            random_start = randint(0, len(random_document) - 1)
                            if len(random_document[random_start]) >= 1:
                                for j in range(random_start, len(random_document)):
                                    tokens_b.extend(random_document[j])
                                    if len(tokens_b) >= target_b_length:
                                        break
                                break

                    # "Return" ("put the unused segments back") the unused segments
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                elif not random_next_sentence and random() < 0.5:
                    case = 2
                    is_random_next = True
                    # If the a_end - len(current_chunk) <= 1 (i.e., no leftover sentence to swap)
                    # Swap the second to last segment and the last segment
                    if a_end - len(current_chunk) <= 1 and len(current_chunk) > 2:
                        current_chunk[-2], current_chunk[-1] = current_chunk[-1], current_chunk[-2]
                        for j in range(a_end - 1):
                            tokens_a.extend(current_chunk[j])
                        tokens_b.extend(current_chunk.pop())
                    else:
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    # In this case, simply swap `tokens_a` and `tokens_b`
                    # This leverages the immediately following sentence (i.e., text chunks)
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    case = 3
                    # Actual next sentence
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                try:
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                except AssertionError as e:
                    print("AssertionError: {}".format(e))
                    print("random_next_sentence : {}".format(random_next_sentence))
                    print("tokens_a : {}".format(tokens_a))
                    print("tokens_b : {}".format(tokens_b))
                    print("current_chunk : {}".format(current_chunk))
                    print("case : {}".format(str(case)))
                    print("\n{}".format(all_documents[random_document_index]))
                    exit()

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)

                tokens.append("[SEP]")
                segment_ids.append(1)

                if model_prefix == "bert":
                    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions_bert(
                        tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)
                    instance = {
                        "tokens": tokens,
                        "segment_ids": segment_ids,
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_labels": masked_lm_labels}

                elif model_prefix == "albert":
                    tokens, masked_lm_positions, masked_lm_labels, token_boundary = create_masked_lm_predictions_albert(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_list
                    )
                    # Label for sentence order prediction (sop) in Albert
                    # `sop_label = 1` the sentence order is reversed.
                    # `sop_label = 0` the sentence order is unchanged.
                    sop_label = 1 if is_random_next else 0

                    instance = {
                        "tokens": tokens,
                        "segment_ids": segment_ids,
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_labels": masked_lm_labels,
                        "sop_label": sop_label}

                instances.append(instance)
            # reset and start new chunk
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def truncate_seq(tokens, max_num_tokens):
    """Truncates a list to a maximum sequence length"""
    while len(tokens) > max_num_tokens:
        assert len(tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def split_digits(wps, bert_model="bert", subword=True):
    # further split numeric wps
    pattern = re.compile(r"\d+([\d,.]+)?\d*")  # Deal with numbers like "7,000", "0.159"
    toks = []
    if bert_model == "bert":
        for wp in wps:
            if len(pattern.findall(wp)) > 0:
                wp = re.sub(r"[,.]", "", wp)
            if set(wp).issubset(set('#0123456789')) and set(wp) != {'#'}:  # numeric wp - split digits
                for i, dgt in enumerate(list(wp.replace('#', ''))):
                    prefix = '##' if (wp.startswith('##') or i > 0) and subword else ''
                    toks.append(prefix + dgt)
            else:
                toks.append(wp)
    elif bert_model == "roberta":
        # Further split numeric wps by Byte-Pair Encoding as in RoBERTa (e.g., Ġ (\u0120) in front of the start of every word)
        for wp in wps:
            if len(pattern.findall(wp)) > 0:
                wp = re.sub(r"[,.]", "", wp)
            if set(wp).issubset(set('0123456789\u0120')) and set(wp) != {'\u0120'}:
                for i, dgt in enumerate(list(wp.replace('\u0120', ''))):
                    prefix = '\u0120' if (wp.strip()[0] == '\u0120' and i == 0 and subword) else ''
                    toks.append(prefix + dgt)
            else:
                toks.append(wp)
    elif bert_model == "albert":
        for wp in wps:
            if len(pattern.findall(wp)) > 0:
                wp = re.sub(r"[,.]", "", wp)
            if set(wp).issubset(set('0123456789▁')) and set(wp) != {'▁'}:  # Special '▁' token (not an underscore!)
                for i, dgt in enumerate(list(wp.replace('▁', ''))):
                    prefix = '▁' if (wp.strip()[0] == '▁' and i == 0 and subword) else ''
                    toks.append(prefix + dgt)
            else:
                toks.append(wp)
    else:
        raise TypeError("The `bert_model` should be one of bert, roberta and albert.")
    return toks


def split_digits_nonsubwords(wps):
    # Further split numeric wps - but remove "##" (the subword indicator)
    toks = []
    for wp in wps:
        if set(wp).issubset(set('\u0120▁#0123456789')) and not set(wp).issubset({'\u0120', '▁', '#'}):
            new_wp = wp.replace('#', '').replace('▁', '').replace('\u0120', '')
            for i, dgt in enumerate(list(new_wp)):
                toks.append(dgt)
        else:
            toks.append(wp)
    return toks


def main():
    parser = ArgumentParser(description='''Creates whole-word-masked instances for MLM task. MLM_paras.jsonl is a list of dicts each with a key 'sents' and val a list of sentences of some document.\n
    Usage: python gen_train_data_MLM.py --train_corpus MLM_paras.jsonl --bert_model bert-base-uncased --output_dir data/MLM_train/ --do_lower_case --max_predictions_per_seq 65 --do_whole_word_mask --digitize ''')
    
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese", "roberta-base", "roberta-large", "albert-xxlarge-v2"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--digitize", action="store_true",
                        help="Whether to further split a numeric wp into digits.")
    parser.add_argument("--epochs_to_generate", type=int, default=1,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=65,
                        help="Maximum number of tokens to mask in each sequence")
    
    args = parser.parse_args()

    model_prefix = args.bert_model.split("-")[0].strip()
    if model_prefix == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    elif model_prefix == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)
    elif model_prefix == "albert":
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_model)
    else:
        raise AttributeError("Specified attribute {} is not found".format(args.bert_model))
    
    make_subword = False
    digit_tokenize = lambda s: split_digits(tokenizer.tokenize(s), bert_model=model_prefix, subword=make_subword)
    try:
        vocab_list = list(tokenizer.vocab.keys())
    except AttributeError:  # Caused by the missing `vocab` attribute from the updated tokenizer
        vocab_list = list(tokenizer.get_vocab().keys())
   
    with jsonlines.open(args.train_corpus, 'r') as reader:
        data = [d for d in tqdm(reader.iter())]
    docs = []
    for d in tqdm(data):
        doc = [digit_tokenize(sent) if args.digitize else tokenizer.tokenize(sent)
               for sent in d['sents']]
        if doc:
            docs.append(doc)

    # docs is a list of docs - each doc is a list of sents - each sent is list of tokens
    args.output_dir.mkdir(exist_ok=True)
    for epoch in trange(args.epochs_to_generate, desc="Epoch"):
        epoch_filename = args.output_dir / f"epoch_{epoch}.jsonl"
        num_instances = 0
        with epoch_filename.open('w') as epoch_file:
            for doc_idx in trange(len(docs), desc="Document"):
                doc_instances = create_instances_from_document(model_prefix,
                    docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                    masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                    whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list)
                for instance in doc_instances:
                    epoch_file.write(json.dumps(instance) + '\n')
                    num_instances += 1
        metrics_file = args.output_dir / f"epoch_{epoch}_metrics.jsonl"
        with metrics_file.open('w') as metrics_file:
            metrics = {
                "num_training_examples": num_instances,
                "max_seq_len": args.max_seq_len
            }
            metrics_file.write(json.dumps(metrics))


if __name__ == '__main__':
    main()

'''python gen_train_data_MLM.py --train_corpus ./data/MLM_paras.jsonl --bert_model bert-base-uncased --output_dir ./data/MLM_train/ --do_lower_case --max_predictions_per_seq 65 --digitize'''
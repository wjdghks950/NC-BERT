import os, json, copy, string, itertools, logging, pickle, argparse, jsonlines
import numpy as np
import re
import random
import torch
from typing import Any, Dict, List, Tuple, Callable
from collections import defaultdict

from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import IGNORED_TOKENS, STRIPPED_CHARACTERS
from tqdm import tqdm

# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer
from w2n import word_to_num
import stanza

# Add the special tokens
CONTEXT_TOKEN = '[CTX]'
CONTEXT_END = '[/CTX]'

# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens("e")
tokenizer.add_tokens(CONTEXT_TOKEN)

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

START_TOK, END_TOK, SPAN_SEP, IGNORE_IDX, MAX_DECODING_STEPS = '@', '\\', ';', 0, 20


class DropExample(object):
    def __init__(self,
                 qas_id,
                 passage_id,
                 question_tokens,
                 passage_tokens,
                 orig_question_tokens,
                 orig_passage_tokens,
                 numbers_in_passage=None,
                 number_indices=None,
                 numbers_in_question=None,
                 q_number_indices=None,
                 num_ent_indices=None,
                 answer_type=None,
                 number_of_answer=None,
                 passage_spans=None,
                 question_spans=None,
                 answer_annotations=None,
                 answer_texts=None
                 ):
        self.qas_id = qas_id
        self.passage_id = passage_id
        self.question_tokens = question_tokens
        self.passage_tokens = passage_tokens
        self.orig_question_tokens = orig_question_tokens
        self.orig_passage_tokens = orig_passage_tokens
        self.numbers_in_passage = numbers_in_passage
        self.number_indices = number_indices
        self.numbers_in_question = numbers_in_question
        self.q_number_indices = q_number_indices
        self.num_ent_indices = num_ent_indices
        self.answer_type = answer_type
        self.number_of_answer = number_of_answer
        self.passage_spans = passage_spans
        self.question_spans = question_spans
        self.answer_annotations = answer_annotations
        self.answer_texts = answer_texts

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", \nquestion: %s" % (" ".join(self.question_tokens))
        s += ", \npassage: %s" % (" ".join(self.passage_tokens))
        if self.numbers_in_passage:
            s += ", \nnumbers_in_passage: {}".format(self.numbers_in_passage)
        if self.number_indices:
            s += ", \nnumber_indices: {}".format(self.number_indices)
        if self.numbers_in_question:
            s += ", \nnumbers_in_question: {}".format(self.numbers_in_question)
        if self.q_number_indices:
            s += ", \nq_number_indices: {}".format(self.q_number_indices)
        if self.num_ent_indices:
            s += ", \nnum_ent_indices: {}".format(self.num_ent_indices)
        if self.local_ctx_tokens:
            s += ", \nlocal_ctx_tokens: {}".format(self.local_ctx_tokens)
        if self.answer_type:
            s += ", \nanswer_type: {}".format(self.answer_type)
        if self.number_of_answer:
            s += ", \nnumber_of_answer: {}".format(self.number_of_answer)
        if self.passage_spans:
            s += ", \npassage_spans: {}".format(self.passage_spans)
        if self.question_spans:
            s += ", \nquestion_spans: {}".format(self.question_spans)
        if self.answer_annotations:
            s += ", \nanswer_annotations: {}".format(self.answer_annotations)
        if self.answer_type:
            s += ", \nanswer_type: {}".format(self.answer_type)
        if self.answer_texts:
            s += ", \nanswer_texts: {}".format(self.answer_texts)
        return s


class DropFeatures(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 max_seq_length,
                 tokens,
                 que_token_to_orig_map,
                 doc_token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 numbers_in_input,
                 number_indices,
                 digit_ent_indices=None,
                 digit_type_indices=None,
                 digit_pos=None,
                 start_indices=None,
                 end_indices=None,
                 number_of_answers=None,
                 decoder_label_ids=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.max_seq_length = max_seq_length
        self.tokens = tokens
        self.que_token_to_orig_map = que_token_to_orig_map
        self.doc_token_to_orig_map = doc_token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.numbers_in_input = numbers_in_input
        self.number_indices = number_indices
        self.digit_ent_indices = digit_ent_indices
        self.digit_type_indices = digit_type_indices
        self.digit_pos = digit_pos
        self.start_indices = start_indices
        self.end_indices = end_indices
        self.number_of_answers = number_of_answers
        self.decoder_label_ids = decoder_label_ids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "unique_id: %s" % (self.unique_id)
        s += ", \nnumber_indices: {}".format(self.number_indices)
        if self.start_indices:
            s += ", \nstart_indices: {}".format(self.start_indices)
        if self.end_indices:
            s += ", \nend_indices: {}".format(self.end_indices)
        if self.number_of_answers:
            s += ", \nnumber_of_answers: {}".format(self.number_of_answers)
        return s


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset, ent_type_=token.ent_type_))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset, ent_type_=token.ent_type_))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]


def split_tokens_by_hyphen(tokens: List[Token], sentence_id: List[int] = None) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for i, token in enumerate(tokens):
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                        if sentence_id is not None and token.text != "-":
                            sentence_id[i:i] = [sentence_id[i]] * (len(split_tokens) - 1)  # List assignment to insert multiple `sentence_id`s
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)
    if sentence_id is None:
        return new_tokens
    else:
        return new_tokens, sentence_id


class DropReader(object):
    def __init__(self,
                 max_n_samples: int = -1,
                 include_more_numbers: bool = False,
                 max_number_of_answer: int = 8,
                 include_multi_span: bool = True,
                 preprocess_type = None,
                 logger = None
                 ) -> None:
        super().__init__()
        self._tokenizer = WordTokenizer()
        self.include_more_numbers = include_more_numbers
        self.max_number_of_answer = max_number_of_answer
        self.include_multi_span = include_multi_span
        self.preprocess_type = preprocess_type
        self.logger = logger
        self.max_n_samples = max_n_samples

    def _read(self, file_path: str):
        self.logger.info('creating examples')
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        self.logger.info("Reading file at %s", file_path)
        dataset = read_file(file_path)
        examples, skip_count = [], 0
        for i, (passage_id, passage_info) in enumerate(tqdm(dataset.items())):
            passage_text = passage_info["passage"]
            passage_tokens = None
            sentence_id = None
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]
                if not answer_annotations:
                    # for test set use a dummy
                    answer_annotations = [{'number':'0', 'date':{"month":'', "day":'', "year":''}, 'spans':[]}]

                try:
                    example, passage_tokens, sentence_id = self.text_to_example(question_text, passage_text, passage_id, question_id, answer_annotations, passage_tokens, sentence_id)
                except TypeError as e:
                    print(e)
                    
                if example is not None:
                    examples.append(example)
                else:
                    skip_count += 1
            if self.max_n_samples > 0 and len(examples) >= self.max_n_samples:
                break
        self.logger.info(f"Skipped {skip_count} examples, kept {len(examples)} examples.")
        return examples
    
    def text_to_example(self,  # type: ignore
                        question_text: str,
                        passage_text: str,
                        passage_id: str,
                        question_id: str,
                        answer_annotations: List[Dict] = None,
                        passage_tokens: List[Token] = None,
                        sentence_id: List[Token] = None):

        # Use Stanza to extract NER
        bio_pattern = re.compile(r"[A-Z]-")

        if not passage_tokens and not sentence_id:
            passage_doc = nlp(passage_text)  # Use `Stanza` to tokenize text
            passage_tokens = []
            sentence_id = []
            for i, sent in enumerate(passage_doc.sentences):
                for word in sent.tokens:
                    word_ner = bio_pattern.sub('', word.ner)
                    passage_tokens.append(Token(text=word.text, idx=word.start_char, ent_type_=word_ner))  # `w.start_char` in Stanza is `w.idx` in allennlp Token
                    sentence_id.append(i)  # Store `sentence_id` for entity-number connection
            passage_tokens, sentence_id = split_tokens_by_hyphen(passage_tokens, sentence_id)

            # `sentence_id` only pertains to `passage_tokens`
            assert len(passage_tokens) == len(sentence_id)
            
        question_doc = nlp(question_text)
        question_tokens = []
        for sent in question_doc.sentences:
            for word in sent.tokens:
                word_ner = bio_pattern.sub('', word.ner)
                question_tokens.append(Token(text=word.text, idx=word.start_char, ent_type_=word_ner))
        question_tokens = split_tokens_by_hyphen(question_tokens)

        orig_passage_tokens = copy.deepcopy(passage_tokens)
        orig_question_tokens = copy.deepcopy(question_tokens)

        answer_type: str = None
        answer_texts: List[str] = []
        number_of_answer: int = None
        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually this doesn't affect
            # the training, because we only have one annotation for the train set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])
            number_of_answer = min(len(answer_texts), self.max_number_of_answer)
        
        if answer_type is None or (answer_type == 'spans' and len(answer_texts) > 1 and not self.include_multi_span): 
            return None  # multi-span
        
        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            # answer_tokens = self._tokenizer.tokenize(answer_text)
            # answer_tokens = split_tokens_by_hyphen(answer_tokens)
            answer_doc = nlp(answer_text)
            answer_tokens = [Token(text=w.text, idx=w.start_char) for w in answer_doc.iter_words()]
            answer_tokens = split_tokens_by_hyphen(answer_tokens)
            tokenized_answer_texts.append(answer_tokens)

        # During our pre-training, we dont handle "12,000", etc and hence
        # are normalizing the numbers in passage, question and answer texts.
        def _normalize_number_tokens(tokens: List[Token], is_answer_text=False) -> List[Token]:
            # returns a new list of tokens after normalizing the numeric tokens
            new_tokens = []
            for token in tokens:
                number = self.get_number_from_word(token.text)
                if number is not None and self.preprocess_type not in ["surround", "digitctx", "skipctx"]:  # `skipctx` - skipconnect + [CTX] token / `skipconnect` - skipconnect only
                    new_tokens.append(Token(str(number)))
                else:
                    new_tokens.append(token)
            return new_tokens
        passage_tokens = _normalize_number_tokens(passage_tokens)
        question_tokens = _normalize_number_tokens(question_tokens)

        tokenized_answer_texts_list = copy.deepcopy(tokenized_answer_texts)
        tokenized_answer_texts = []
        for answer_tokens in tokenized_answer_texts_list:  # (E.g., `tokenized_answer_texts = [[100, Peso, note], [500, Peso, note]]`)
            tokenized_answer_texts.append(_normalize_number_tokens(answer_tokens, is_answer_text=True))

        normalized_answer_texts = list(map(lambda l: ' '.join([t.text for t in l]), tokenized_answer_texts))
        
        # for multi-span we remove duplicates and arrange by order of occurrence in passage
        unique_answer_texts = sorted(set(normalized_answer_texts), key=normalized_answer_texts.index)
        passage_text = ' '.join([token.text for token in passage_tokens])
        arranged_answer_texts = sorted(unique_answer_texts, key=passage_text.find)
        normalized_answer_texts = arranged_answer_texts
        
        # Store `ents_in_passage` and `ents_indices` - Entities in text
        ents_in_passage = []
        ent_indices = []
        entity_types = ["PERSON", "ORG", "NORP", "FAC", "GPE", "LOC"]

        numbers_in_passage = []
        number_indices = []

        sent2num = defaultdict(list)
        sent2ent = defaultdict(list)
        for token_index, token in enumerate(passage_tokens):  # Save entities and their indices for ent-num channel
            if token.ent_type_ in entity_types:
                ents_in_passage.append(token.text)
                ent_indices.append(token_index)
                sent2ent[sentence_id[token_index]].append(token_index)

            number = self.get_number_from_word(token.text)
            if number is not None:  # Save number in the passage for the three channels
                numbers_in_passage.append(number)
                number_indices.append(token_index)
                sent2num[sentence_id[token_index]].append(token_index)
        assert len(numbers_in_passage) == len(number_indices)

        num_ent_indices = None
        if self.preprocess_type in "localattn":
            # To figure out whether a number and entity are in the same sentence,
            # Link every number index to the corresponding entity indices
            num_ent_indices = defaultdict(list)  # Number to Entity index in passage
            for sent_id, num_indices in sent2num.items():
                for num_idx in num_indices:
                    num_ent_indices[num_idx] = sent2ent[sent_id]

            # # 1. A number should attend only to the closest entity
            # num_ent_indices = defaultdict(list)
            # for sent_id, num_indices in sent2num.items():
            #     for num_idx in num_indices:
            #         if len(sent2ent[sent_id]) > 0:
            #             num_ent_indices[num_idx].append(min(sent2ent[sent_id], key=lambda x: abs(x - num_idx)))

        numbers_in_question = []
        q_number_indices = []
        for token_index, token in enumerate(question_tokens):  # Save numbers within the question for later (For DiceEmbedding)
            number = self.get_number_from_word(token.text)
            if number is not None:
                numbers_in_question.append(number)
                q_number_indices.append(token_index)
        
        assert len(numbers_in_question) == len(q_number_indices)

        # print("passage >> ", passage_tokens)
        # print("numbers_in_passage: ", numbers_in_passage)
        # print("number_indices: ", number_indices)

        # print("ents_in_passage: ", ents_in_passage)
        # print("ents_indices: ", ent_indices)

        # print("num_ent_indices : ", num_ent_indices)

        # exit()

        valid_passage_spans = \
            self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
        valid_question_spans = \
            self.find_valid_spans(question_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
        number_of_answer = None if valid_passage_spans == [] and valid_question_spans == [] else number_of_answer

        # print("passage_tokens (in) : ", passage_tokens)
        # print("sentence_id (in) : ", sentence_id)

        example = DropExample(
            qas_id=question_id,
            passage_id=passage_id,
            question_tokens=[token.text for token in question_tokens],
            passage_tokens=[token.text for token in passage_tokens],
            orig_question_tokens=[token.text for token in orig_question_tokens],
            orig_passage_tokens=[token.text for token in orig_passage_tokens],
            numbers_in_passage=numbers_in_passage,
            number_indices=number_indices,
            numbers_in_question=numbers_in_question,
            q_number_indices=q_number_indices,
            num_ent_indices=num_ent_indices,
            answer_type=answer_type,
            number_of_answer=number_of_answer,
            passage_spans=valid_passage_spans,
            question_spans=valid_question_spans,
            answer_annotations=answer_annotations,
            answer_texts=normalized_answer_texts)

        return example, passage_tokens, sentence_id


    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key]
                           for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
            answer_texts = [' '.join(date_tokens)]
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def get_number_from_word(word: str, improve_number_extraction=True):
        punctuation = string.punctuation.replace('-', '')
        word = word.strip(punctuation)
        word = word.replace(",", "")
        if word in ["hundred", "thousand", "million", "billion", "trillion"]:
            return None
        try:
            number = word_to_num(word)
        except ValueError:
            try:
                number = int(word)
            except ValueError:
                try:
                    number = float(word)
                except ValueError:
                    if improve_number_extraction:
                        if re.match('^\d*1st$', word):  # ending in '1st'
                            number = int(word[:-2])
                        elif re.match('^\d*2nd$', word):  # ending in '2nd'
                            number = int(word[:-2])
                        elif re.match('^\d*3rd$', word):  # ending in '3rd'
                            number = int(word[:-2])
                        elif re.match('^\d+th$', word):  # ending in <digits>th
                            # Many occurrences are when referring to centuries (e.g "the *19th* century")
                            number = int(word[:-2])
                        elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                            # Decades, e.g. "1960s".
                            # Other sequences of digits ending with s (there are 39 of these in the training
                            # set), do not seem to be arithmetically related, as they are usually proper
                            # names, like model numbers.
                            number = int(word[:-1])
                        elif len(word) > 2 and re.match(r'^\d+\w[mM]?', word):
                            # km, cm, mm, e.g. "20km", "15cm", or "3M"
                            number = int(re.sub("\D", "", word))
                        elif len(word) > 4 and re.match(r'^\d+(\.?\d+)?/km[²2]$', word):
                            # per square kilometer, e.g "73/km²" or "3057.4/km2"
                            if '.' in word:
                                number = float(word[:-4])
                            else:
                                number = int(word[:-4])
                        elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                            # per month, e.g "1050.95/month"
                            if '.' in word:
                                number = float(word[:-6])
                            else:
                                number = int(word[:-6])
                        else:
                            return None
                    else:
                        return None

        return number

    @staticmethod
    def convert_word_to_number(word: str, try_to_include_more_numbers=False, normalized_tokens=None, token_index=None):
        """
        Currently we only support limited types of conversion.
        """
        if try_to_include_more_numbers:
            # strip all punctuations from the sides of the word, except for the negative sign
            punctuations = string.punctuation.replace('-', '')
            word = word.strip(punctuations)
            # some words may contain the comma as deliminator
            word = word.replace(",", "")
            # word2num will convert hundred, thousand ... to number, but we skip it.
            if word in ["hundred", "thousand", "million", "billion", "trillion"]:
                return None
            try:
                number = word_to_num(word)
            except ValueError:
                try:
                    number = int(word)
                except ValueError:
                    try:
                        number = float(word)
                    except ValueError:
                        number = None
            return number
        else:
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(no_comma_word)
                except ValueError:
                    number = None
            return number

    @staticmethod
    def find_valid_spans(passage_tokens: List[Token],
                         answer_texts: List[List[Token]]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in answer_text]
            num_answer_tokens = len(answer_tokens)
            try:
                if answer_tokens[0] not in word_positions:
                    continue
            except IndexError:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans


def split_digits(wps: List[str]) -> List[str]:
    # Further split numeric wps
    toks = []
    for wp in wps:
        if set(wp).issubset(set('#0123456789')) and set(wp) != {'#'}:  # numeric wp - split digits
            for i, dgt in enumerate(list(wp.replace('#', ''))):
                prefix = '##' if (wp.startswith('##') or i > 0) else ''
                toks.append(prefix + dgt)
        else:
            toks.append(wp)
    return toks


def split_digits_nonsubwords(wps: List[str]) -> List[str]:
    # Further split numeric wps - but remove "##" (the subword indicator)
    toks = []
    for wp in wps:
        if set(wp).issubset(set('#0123456789')) and set(wp) != {'#'}:
            for i, dgt in enumerate(list(wp.replace('#', ''))):
                toks.append(dgt)
        else:
            toks.append(wp)
    return toks


def convert_answer_spans(spans, orig_to_tok_index, all_len, all_tokens):
    tok_start_positions, tok_end_positions = [], []
    for span in spans:
        start_position, end_position = span[0], span[1]
        tok_start_position = orig_to_tok_index[start_position]
        if end_position + 1 >= len(orig_to_tok_index):
            tok_end_position = all_len - 1
        else:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        if tok_start_position < len(all_tokens) and tok_end_position < len(all_tokens):
            tok_start_positions.append(tok_start_position)
            tok_end_positions.append(tok_end_position)
    return tok_start_positions, tok_end_positions


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_decoding_steps, indiv_digits=True, preprocess_type=None, logger=None):
    """Loads a data file into a list of `InputBatch`s."""

    logger.info('Creating features from `examples (DropExample -> DropFeatures)`')
    unique_id = 1000000000
    skip_count, truncate_count = 0, 0
    
    tokenize = (lambda s: split_digits_nonsubwords(tokenizer.tokenize(s))) if indiv_digits else tokenizer.tokenize
    
    features, all_qp_lengths = [], []
    for (example_index, example) in enumerate(tqdm(examples)):
        que_tok_to_orig_index = []
        que_orig_to_tok_index = []
        all_que_tokens = []
        for (i, token) in enumerate(example.question_tokens):
            que_orig_to_tok_index.append(len(all_que_tokens))
            sub_tokens = tokenize(token)  # Further tokenize tokens using transformers tokenizer (e.g., BertTokenizer)
            que_tok_to_orig_index += [i] * len(sub_tokens)
            all_que_tokens += sub_tokens

        # List up all the entities
        if example.num_ent_indices is not None:
            ent_num_indices = defaultdict(list)
            for num_idx, ent_idx in example.num_ent_indices.items():
                for ei in ent_idx:
                    ent_num_indices[ei].append(num_idx)
        
        doc_tok_to_orig_index = []
        doc_orig_to_tok_index = []
        all_doc_tokens = []
        num_subent_indices = defaultdict(list)  # Number (not yet sub-tokenized) to entity (sub-tokenized) index
        for (i, token) in enumerate(example.passage_tokens):
            doc_orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenize(token)
            # Change whole word entities into sub-word entities
            if preprocess_type == "localattn" and example.num_ent_indices is not None and i in ent_num_indices.keys():
                for num_idx in ent_num_indices[i]:
                    num_idx = num_idx if isinstance(num_idx, list) else [num_idx]
                    for ni in num_idx:
                        # (Number index) to (subword tokenized entity index)
                        num_subent_indices[ni] += list(range(len(all_doc_tokens), len(all_doc_tokens) + len(sub_tokens)))

            doc_tok_to_orig_index += [i] * len(sub_tokens)
            all_doc_tokens += sub_tokens

        original_passage_length = len(all_doc_tokens)
            
        # The -3 accounts for [CLS], [SEP] and [SEP]
        # Truncate the passage according to the max sequence length
        max_tokens_for_doc = max_seq_length - len(all_que_tokens) - 3
        all_doc_len = len(all_doc_tokens)
        if all_doc_len > max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[:max_tokens_for_doc]
            truncate_count += 1
        all_qp_lengths.append(len(all_que_tokens) + all_doc_len + 3)
        
        query_tok_start_positions, query_tok_end_positions = \
            convert_answer_spans(example.question_spans, que_orig_to_tok_index, len(all_que_tokens), all_que_tokens)

        passage_tok_start_positions, passage_tok_end_positions = \
            convert_answer_spans(example.passage_spans, doc_orig_to_tok_index, all_doc_len, all_doc_tokens)

        tok_q_number_indices = []
        tok_number_indices = []
        tok_q_digit_pos = []  # `digit_pos` for every single digit (for calculating the dice_loss)
        tok_digit_pos = []

        if preprocess_type in ["localattn", "numeric", "textual"]:
            # These is a list of every index of numeric wps of original number token (example.q_number_indices) - `question`
            assert len(example.numbers_in_question) == len(example.q_number_indices)

            for q_index, q_num in zip(example.q_number_indices, example.numbers_in_question):
                if q_index != -1:
                    q_num_subtoken = tokenize(str(q_num))
                    tok_index = que_orig_to_tok_index[q_index]
                    subtoken_indices = list(range(tok_index, tok_index + len(q_num_subtoken)))
                    for i, subtok_index in enumerate(subtoken_indices):
                        if subtok_index < len(all_que_tokens):
                            tok_q_number_indices.append(subtok_index)
                            tok_q_digit_pos.append(len(subtoken_indices) - 1 - i)  # Adding "digit position" (in exponent) for every number (for every digit)
                else:
                    tok_q_number_indices.append(-1)
                    tok_q_digit_pos.append(-1)

            # These is a list of every index of numeric wps of original number token (example.number_indices) - `passage`
            assert len(example.numbers_in_passage) == len(example.number_indices)
            assert len(tok_q_number_indices) == len(tok_q_digit_pos)

            digit_ent_indices = defaultdict(list)  # digit-level index as key / subword tokenized entity index as value
            digit_type_indices = defaultdict(list)  # digit-level index as key / surrounding "type-defining" words as value (window size = k)
            window_size = 2  # Defining the scope of the surrounding words to look at (number type definition)

            for index, num in zip(example.number_indices, example.numbers_in_passage):
                if index != -1:
                    num_subtoken = tokenize(str(num))
                    tok_index = doc_orig_to_tok_index[index]
                    subtoken_indices = list(range(tok_index, tok_index + len(num_subtoken)))
                    for i, subtok_index in enumerate(subtoken_indices):
                        if subtok_index < len(all_doc_tokens):
                            tok_number_indices.append(subtok_index)
                            # `digit_ent_indices` - mapping between digit-level index to entity indices
                            if index in num_subent_indices.keys():
                                digit_ent_indices[subtok_index] += num_subent_indices[index]
                                digit_type_indices[subtok_index] += list(range(subtoken_indices[0] - window_size, subtoken_indices[0])) + list(range(subtoken_indices[-1] + 1, subtoken_indices[-1] + window_size + 1))
                            # Adding "digit position" for every number (for every digit)
                            tok_digit_pos.append(len(subtoken_indices) - 1 - i)
                else:
                    tok_number_indices.append(-1)
                    tok_digit_pos.append(-1)

            assert len(tok_number_indices) == len(tok_digit_pos)

        else:
            # These are the _starting_indices of numeric wps of original number token (example.q_number_indices) - `question`
            for q_index in example.q_number_indices:
                if q_index != -1:
                    tok_index = que_orig_to_tok_index[q_index]
                    if tok_index < len(all_que_tokens):
                        tok_q_number_indices.append(tok_index)
                else:
                    tok_q_number_indices.append(-1)

            # These are the _starting_ indices of numeric wps of original number token (example.number_indices) - `passage`
            for index in example.number_indices:
                if index != -1:
                    tok_index = doc_orig_to_tok_index[index]
                    if tok_index < len(all_doc_tokens):
                        tok_number_indices.append(tok_index)
                else:
                    tok_number_indices.append(-1)
        
        # print("doc_tok_to_orig_index: ", doc_tok_to_orig_index)
        # print("doc_orig_to_tok_index: ", doc_orig_to_tok_index)
        # print("all_que_tokens: ", all_que_tokens)
        # print("all_que_tokens (length): ", len(all_que_tokens))
        # print("all_doc_tokens: ", all_doc_tokens)
        # print("all_doc_tokens (length): ", len(all_doc_tokens))
        # print("(original) number_indices: ", example.number_indices)
        # print("tok_q_number_indices: ", tok_q_number_indices)
        # print("tok_q_digit_pos: ", tok_q_digit_pos)
        # print("tok_digit_pos: ", tok_digit_pos)
        # print("LENGTH: {} / {}".format(len(example.number_indices), len(tok_number_indices)))

        # exit()

        if original_passage_length == len(all_doc_tokens) and preprocess_type not in ["numeric", "textual", "localattn"]:
            assert len(example.number_indices) == len(tok_number_indices)  # Assert only when the numbers are split into "digit-level"

        tokens, segment_ids = [], []
        que_token_to_orig_map = {}
        doc_token_to_orig_map = {}
        tokens.append("[CLS]")
        for i in range(len(all_que_tokens)):
            que_token_to_orig_map[len(tokens)] = que_tok_to_orig_index[i]
            tokens.append(all_que_tokens[i])
        tokens.append("[SEP]")
        segment_ids += [0] * len(tokens)

        for i in range(len(all_doc_tokens)):
            doc_token_to_orig_map[len(tokens)] = doc_tok_to_orig_index[i]
            tokens.append(all_doc_tokens[i])
        tokens.append("[SEP]")
        segment_ids += [1] * (len(tokens) - len(segment_ids))
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # segment_ids are 0 for toks in [CLS] Q [SEP] and 1 for P [SEP]
        assert len(segment_ids) == len(input_ids)
        
        # we expect the generative head to output the wps of the joined answer text
        answer_text = (SPAN_SEP + ' ').join(example.answer_texts).strip()

        dec_toks = [START_TOK] + tokenize(answer_text) + [END_TOK]
        # ending the number with a '\\' to signify end
        dec_toks = dec_toks[:max_decoding_steps]
        dec_ids = tokenizer.convert_tokens_to_ids(dec_toks)
        decoder_label_ids = dec_ids + [IGNORE_IDX]*(max_decoding_steps - len(dec_ids))
        # IGNORE_IDX is ignored while computing the loss
        
        # print("answer_text: ", answer_text)
        # print("dec_toks: ", dec_toks)
        # print("dec_ids: ", dec_ids)

        q_number_indices = []  # again these are the starting indices of wps of an orig number token - `question`
        que_offset = 1
        for tok_q_number_index in tok_q_number_indices:
            if tok_q_number_index != -1:
                q_number_index = tok_q_number_index + que_offset
                q_number_indices.append(q_number_index)
            else:
                q_number_indices.append(-1)
        
        numbers_in_question = example.numbers_in_question[:len(q_number_indices)]  # The actual numbers extracted from the question

        number_indices = []  # again these are the starting indices of wps of an orig number token - `passage`
        doc_offset = len(all_que_tokens) + 2
        for tok_number_index in tok_number_indices:
            if tok_number_index != -1:
                number_index = tok_number_index + doc_offset
                number_indices.append(number_index)
            else:
                number_indices.append(-1)
        
        numbers_in_passage = example.numbers_in_passage[:len(number_indices)]  # The actual numbers extracted from the passage

        # Reflect the index shift in the number indices in `digit_ent_indices` and entity indices in `digit_ent_indices` with `doc_offset`
        new_digit_ent_indices = defaultdict(list)
        for digit_idx, ent_idx in digit_ent_indices.items():
            # ent_idx = ent_idx if isinstance(ent_idx, list) else [ent_idx]
            if digit_idx != -1 and digit_idx + doc_offset < max_seq_length:
                new_digit_ent_indices[digit_idx + doc_offset] += [i + doc_offset for i in ent_idx if i + doc_offset < max_seq_length - 1]

        new_digit_type_indices = defaultdict(list)
        for digit_idx, type_idx in digit_type_indices.items():
            # type_idx = type_idx if isinstance(type_idx, list) else [type_idx]
            if digit_idx != -1 and digit_idx + doc_offset < max_seq_length:
                new_digit_type_indices[digit_idx + doc_offset] += [i + doc_offset for i in type_idx if i + doc_offset < max_seq_length - 1]

        if preprocess_type not in ["numeric", "textual", "localattn"]:
            assert len(numbers_in_question) == len(q_number_indices)
            assert len(numbers_in_passage) == len(number_indices)

        # Combine the numbers in both question and passage
        numbers_in_input = numbers_in_question + numbers_in_passage
        number_indices = q_number_indices + number_indices
        digit_pos = tok_q_digit_pos + tok_digit_pos

        assert len(number_indices) == (len(tok_q_digit_pos) + len(tok_digit_pos))

        # print("\nINPUT: ", tokens)
        # print("\nnumbers_in_question: ", numbers_in_question)
        # print("q_number_indices\n: ", q_number_indices)

        # print("\nnumbers_in_passage: ", numbers_in_passage)
        # print("number_indices\n: ", number_indices)
        # print("combined_numbers (que + passage) : ", numbers_in_question + numbers_in_passage)
        # print("combined_numbers (que + passage) (index) : ", number_indices)

        # print("tok_number_indices: ", tok_number_indices)
        # print("digit_pos: ", digit_pos)
        # print("==" * 50)

        # print("new_digit_ent_indices : ", new_digit_ent_indices)
        # for digit_idx, ent_idx in new_digit_ent_indices.items(): 
        #     print("digit_ent_mask : ", digit_ent_mask[digit_idx])
        #     print("ent_idx : ", ent_idx)
        # exit()

        start_indices, end_indices, number_of_answers = [], [], []
        # Shift the answer span indices according to [CLS] Q [SEP] P [SEP] structure
        if passage_tok_start_positions != [] and passage_tok_end_positions != []:
            for tok_start_position, tok_end_position in zip(passage_tok_start_positions, 
                                                            passage_tok_end_positions):
                start_indices.append(tok_start_position + doc_offset)
                end_indices.append(tok_end_position + doc_offset)
        
        if query_tok_start_positions != [] and query_tok_end_positions != []:
            for tok_start_position, tok_end_position in zip(query_tok_start_positions, 
                                                            query_tok_end_positions):
                start_indices.append(tok_start_position + que_offset)
                end_indices.append(tok_end_position + que_offset)

        if start_indices != [] and end_indices != []:
            assert example.number_of_answer is not None
            number_of_answers.append(example.number_of_answer - 1)

        features.append(DropFeatures(
                unique_id=unique_id,
                example_index=example_index,
                max_seq_length=max_seq_length,
                tokens=tokens,
                que_token_to_orig_map=que_token_to_orig_map,
                doc_token_to_orig_map=doc_token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                numbers_in_input=numbers_in_input,
                number_indices=number_indices,
                digit_ent_indices=new_digit_ent_indices,
                digit_type_indices=new_digit_type_indices,
                digit_pos=digit_pos,
                start_indices=start_indices,
                end_indices=end_indices,
                number_of_answers=number_of_answers,
                decoder_label_ids=decoder_label_ids))
        unique_id += 1

    logger.info(f"Skipped {skip_count} features, truncated {truncate_count} features, kept {len(features)} features.")
    return features, all_qp_lengths


def read_file(file):
    if file.endswith('jsonl'):
        with jsonlines.open(file, 'r') as reader:
            return [d for d in reader.iter()]
    
    if file.endswith('json'):
        with open(file, encoding='utf8') as f:
            return json.load(f)

    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'rb') as f:
            return pickle.load(f)


def write_file(data, file):
    if file.endswith('jsonl'):
        with jsonlines.open(file, mode='w') as writer:
            writer.write_all(data)

    if file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description='Used to make examples, features from DROP-like dataset.')
    parser.add_argument("--drop_json", default='drop_dataset_dev.json', type=str, help="The eval .json file.")
    parser.add_argument("--split", default='eval', type=str, help="data split: train/eval.")
    parser.add_argument("--max_n_samples", default=-1, type=int, help="max num samples to process.")
    parser.add_argument("--output_dir", default='data/examples_n_features', type=str, help="Output dir.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="max seq len of [cls] q [sep] p [sep]")
    parser.add_argument("--max_decoding_steps", default=20, type=int, help="max tokens to be generated by decoder")
    parser.add_argument("--percent", default="all", type=str, help="Percentage of drop_dataset_train for sample efficiency test.")
    parser.add_argument("--preprocess_type", default="localattn", type=str, help="Keyword for preprocessing type - `localattn`, `numeric`, `textual`")

    args = parser.parse_args()
    
    drop_reader = DropReader(max_n_samples=args.max_n_samples,
                             include_more_numbers=True,
                             max_number_of_answer=8,
                             include_multi_span=True,
                             preprocess_type=args.preprocess_type,
                             logger=logger)
    print("PATH: " , args.drop_json.split("_"))
    examples = drop_reader._read(args.drop_json)

    features, lens = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_decoding_steps=args.max_decoding_steps,
        indiv_digits=True,
        preprocess_type=args.preprocess_type,
        logger=logger)

    logger.info("*** convert_examples_to_features completed ***")

    os.makedirs(args.output_dir, exist_ok=True)
    split = 'train' if args.split == 'train' else 'eval'
    if (args.percent == "all" and args.drop_json.split("_")[2] == "dataset") or args.preprocess_type in ["numeric", "textual"]:
        if args.preprocess_type == "diceloss":
            args.preprocess_type = "dataset"
        else:
            example_path = args.output_dir + '/%s_%s_examples.pkl' % (split, args.preprocess_type)
            feature_path = args.output_dir + '/%s_%s_features.pkl' % (split, args.preprocess_type)

    if args.drop_json.split("_")[3][:-5] in ["add10", "add100", "factor10", "factor100", "randadd100", "randfactor100"]:
        print("Perturbation Type: {}".format(args.drop_json.split("_")[3][:-5]))
        example_path = args.output_dir + '/%s_%s_%s_examples.pkl' % (split, args.preprocess_type, args.drop_json.split("_")[3][:-5])
        feature_path = args.output_dir + '/%s_%s_%s_features.pkl' % (split, args.preprocess_type, args.drop_json.split("_")[3][:-5])

    write_file(examples, example_path)
    write_file(features, feature_path)

    print("Example Pkl file saved at {}.".format(example_path))
    print("Feature Pkl file saved at {}.".format(feature_path))


if __name__ == "__main__":
    main()

'''
# drop  (MAX_DECODING_STEPS = 20)
python create_examples_n_features.py --split train --drop_json ../../data/drop_dataset_train.json --output_dir data/examples_n_features --max_seq_length 512
python create_examples_n_features.py --split eval --drop_json ../../data/drop_dataset_dev.json --output_dir data/examples_n_features --max_seq_length 512

# texual synthetic data:  (MAX_DECODING_STEPS = 20)
python create_examples_n_features.py --split train --drop_json ../../data/synthetic_textual_mixed_min3_max6_up0.7_train_drop_format.json --output_dir data/examples_n_features_syntext --max_seq_length 160 --max_n_samples -1
python create_examples_n_features.py --split eval --drop_json ../../data/synthetic_textual_mixed_min3_max6_up0.7_dev_drop_format.json --output_dir data/examples_n_features_syntext --max_seq_length 160

# numeric data (use MAX_DECODING_STEPS = 11):
python create_examples_n_features.py --split train --drop_json ../../data/synthetic_numeric_train_drop_format.json --output_dir data/examples_n_features_numeric --max_seq_length 50 --max_decoding_steps 11 --max_n_samples -1
python create_examples_n_features.py --split eval --drop_json ../../data/synthetic_numeric_dev_drop_format.json --output_dir data/examples_n_features_numeric --max_seq_length 50 --max_decoding_steps 11

# numeric data without DT (use indiv_digits=False, MAX_DECODING_STEPS = 11):
python create_examples_n_features.py --split train --drop_json ../../data/synthetic_numeric_train_drop_format.json --output_dir data/examples_n_features_numeric_wp --max_seq_length 42 --max_decoding_steps 11 --max_n_samples -1
python create_examples_n_features.py --split eval --drop_json ../../data/synthetic_numeric_dev_drop_format.json --output_dir data/examples_n_features_numeric_wp --max_seq_length 42 --max_decoding_steps 11
'''


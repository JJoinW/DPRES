import codecs
import csv

import nltk
# import logging
import re
import numpy as np
from utils import get_logger
import pandas as pd

url_replacer = '<url>'
logger = get_logger("Loading data...")
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

MAX_SENTLEN = 50
MAX_SENTNUM = 100

asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
    9: (0, 1)
}

trait_ranges = {
    0: (0, 12),
    1: (1, 6),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 6),
    8: (2, 12)
}

num_traits = {
    0: 6,
    1: 5,
    2: 5,
    3: 4,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
    8: 6
}


def get_score_range(prompt_id):
    return asap_ranges[prompt_id]


def get_trait_score_range(prompt_id):
    return trait_ranges[prompt_id]


def get_model_friendly_scores(scores_array, prompt_id):
    arg_type = type(prompt_id)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id]
        scores_array = (scores_array - low) / (high - low)
    return scores_array


def get_trait_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = trait_ranges[prompt_id_array]
        scores_array = (scores_array - low) / (high - low)
    return scores_array


def get_model_and_trait_friendly_scores(scores_array, prompt_id, overall_score_column, attr_score_columns,
                                        original_score_index):
    norm_score = []
    # We are interested to use any traits as primary and as auxiliary task.
    # For that original_score_index = 6, overall_score_column = whichever trait we want to make primary
    # This is for overall score column if it is equal to 6 then we normalize the value according to the asap ranges otherwise trait ranges.
    all_col = [overall_score_column] + attr_score_columns
    all_col = [int(col) for col in all_col]
    # Similarily for other attributes columns but now this columns might contain col 6 so normalize it according to the asap ranges.
    for i, score_array in zip(all_col, scores_array):
        if i == original_score_index:
            norm_score.append(get_model_friendly_scores(score_array, prompt_id))
        else:
            norm_score.append(get_trait_friendly_scores(score_array, prompt_id))

    return norm_score


def unify_training_scores(score, prompt_id, essay_set, score_index, overall_score_column):
    if essay_set == 9:
        if score_index == overall_score_column:
            low, high = asap_ranges[prompt_id]  # this is when score column is considered as STL
        else:
            low, high = trait_ranges[prompt_id]  ##this is when trait columns considered as STL
        new_score = (score * (high - low)) + low
        return np.round(new_score)
    return score


def unify_trait_scores(score, prompt_id, essay_set):
    if essay_set != 9:
        min, max = trait_ranges[prompt_id]
        new_score = (score - min) / (max - min)
        return new_score
    return score


def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = scores_array * (high - low) + low
        assert np.all(scores_array >= low) and np.all(scores_array <= high)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = scores_array * (high - low) + low
    return scores_array


def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    return tokens


def is_number(token):
    return bool(num_regex.match(token))


def read_essays(file_path, prompt_id):
    logger.info('Reading tsv from: ' + file_path)
    essays_list = []
    essays_ids = []
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            if int(tokens[1]) == prompt_id or prompt_id <= 0:
                essays_list.append(tokens[2].strip())
                essays_ids.append(int(tokens[0]))
    return essays_list, essays_ids


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
        # print text
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
        # print text
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)
        # print text

    tokens = tokenize(text)
    if tokenize_sent_flag:
        punctuation = '.!,;:?"\'、，；'
        text = " ".join(tokens)

        # text = text.replace('.', ' [SEP] ')
        # text = text.replace('!', ' [SEP] ')
        # text = text.replace('?', ' [SEP] ')

        text_nopun = re.sub(r'[{}]+'.format(punctuation), '', text)
        sent_tokens = text_nopun
        # sent_tokens = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
        # sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        # print sent_tokens
        # sys.exit(0)
        # if not create_vocab_flag:
        #     print "After processed and tokenized, sentence num = %s " % len(sent_tokens)
        return sent_tokens
    else:
        raise NotImplementedError


def read_dataset(file_path, prompt_id, score_index=6, char_level=False, tokenizer=None):
    logger.info('Reading dataset from: ' + file_path)
    attr_num = num_traits[prompt_id] + 1
    attr_col = [score_index + i for i in range(attr_num)]

    data_x_id, data_y, prompt_ids = [], [], []

    true_score = []
    norm_score = []

    # create lists for attribute scores
    for _ in range(attr_num):
        true_score.append([])
        norm_score.append([])

    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()

            #  read attribute scores
            attr_min_score, attr_max_score = trait_ranges[prompt_id]
            if essay_set == prompt_id or prompt_id <= 0:
                # tokenize text into sentences
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
                if char_level:
                    raise NotImplementedError

                score = float(tokens[score_index])
                true_score_v = score
                true_score[0].append(true_score_v)
                overall_min_score, overall_max_score = asap_ranges[prompt_id]
                norm_score_v = (true_score_v - overall_min_score) / (overall_max_score - overall_min_score)
                norm_score[0].append(norm_score_v)

                for i, col in enumerate(attr_col[1:], 0):
                    if essay_set != 9:
                        true_score_v = unify_trait_scores(float(tokens[int(col)]), prompt_id, essay_set)
                    else:
                        true_score_v = float(tokens[int(col)])
                    true_score[i + 1].append(float(tokens[int(col)]))
                    norm_score[i + 1].append(true_score_v)

                # length = len(sent_tokens)
                # if max_sentnum < length:
                #     max_sentnum = length
                # sent_tokens = '[CLS] ' + sent_tokens

                tokenized_text = tokenizer.tokenize(sent_tokens)
                max_num = 512
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

                data_x_id.append(indexed_tokens)

    prompt_ids.append(essay_set)

    return data_x_id, true_score, norm_score, prompt_ids, max_num


def get_score_vector_positions():
    return {
        1: {'score': 0, 'content': 1, 'organization': 2, 'word choice': 3,
            'sentence fluency': 4, 'conventions': 5},
        2: {'score': 0, 'content': 1, 'organization': 2, 'word choice': 3,
            'sentence fluency': 4, 'conventions': 5},
        3: {'score': 0, 'content': 1, 'prompt adherence': 2, 'language': 3, 'narrativity': 4},
        4: {'score': 0, 'content': 1, 'prompt adherence': 2, 'language': 3, 'narrativity': 4},
        5: {'score': 0, 'content': 1, 'prompt adherence': 2, 'language': 3, 'narrativity': 4},
        6: {'score': 0, 'content': 1, 'prompt adherence': 2, 'language': 3, 'narrativity': 4},
        7: {'score': 0, 'content': 1, 'organization': 2, 'conventions': 3, 'style': 4},
        8: {'score': 0, 'content': 1, 'organization': 2, 'word choice': 3, 'sentence fluency': 4,
            'conventions': 5, 'voice': 6}}


def read_dataset_single(file_path, prompt_id, score_index=7, char_level=False, tokenizer=None, attribute='score'):
    logger.info('Reading dataset from: ' + file_path)

    data_x_id, data_y, prompt_ids = [], [], []

    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()

            if essay_set == prompt_id or prompt_id <= 0:
                # tokenize text into sentences
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
                if char_level:
                    raise NotImplementedError

                score = float(tokens[score_index])
                true_score_v = score
                overall_min_score, overall_max_score = asap_ranges[prompt_id]
                norm_score_v = (true_score_v - overall_min_score) / (overall_max_score - overall_min_score)
                data_y.append(norm_score_v)

                tokenized_text = tokenizer.tokenize(sent_tokens)
                max_num = 512
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

                data_x_id.append(indexed_tokens)

    prompt_ids.append(essay_set)

    return data_x_id, data_y, prompt_ids, max_num


def read_dataset_for_ASAP(prompt_id, file_path, score_index=3, char_level=False, tokenizer=None):
    logger.info('Reading dataset from: ' + file_path)
    attr_num = num_traits[prompt_id] + 1

    prompt_attr_col = {
        1: [score_index + i for i in range(attr_num)],
        2: [score_index + i for i in range(attr_num)],
        3: [score_index + i for i in [0, 1, 6, 7, 8]],
        4: [score_index + i for i in [0, 1, 6, 7, 8]],
        5: [score_index + i for i in [0, 1, 6, 7, 8]],
        6: [score_index + i for i in [0, 1, 6, 7, 8]],
        7: [score_index + i for i in [0, 1, 2, 5, 9]],
        8: [score_index + i for i in [0, 1, 2, 3, 4, 5, 10]]
    }
    attr_col = prompt_attr_col[prompt_id]

    data_x_id, data_y, prompt_ids = [], [], []

    true_score = []
    norm_score = []

    # create lists for attribute scores
    for _ in range(attr_num):
        true_score.append([])
        norm_score.append([])

    LLM_gen = pd.read_csv(file_path)
    for index, row in LLM_gen.iterrows():
        essay_id = row['essay_id']
        essay_set = row['essay_set']
        essay_content = row['essay']

        if essay_set == prompt_id:
            score = float(row.iloc[score_index])
            true_score_v = score
            true_score[0].append(true_score_v)
            overall_min_score, overall_max_score = asap_ranges[prompt_id]
            norm_score_v = (true_score_v - overall_min_score) / (overall_max_score - overall_min_score)
            norm_score[0].append(norm_score_v)

            for i, col in enumerate(attr_col[1:], 0):
                if essay_set != 9:
                    true_score_v = unify_trait_scores(float(row.iloc[col]), prompt_id, essay_set)
                else:
                    true_score_v = float(row.iloc[col])
                true_score[i + 1].append(float(row.iloc[col]))
                norm_score[i + 1].append(true_score_v)

            # tokenize text into sentences
            sent_tokens = text_tokenizer(essay_content, replace_url_flag=True, tokenize_sent_flag=True)
            if char_level:
                raise NotImplementedError

            tokenized_text = tokenizer.tokenize(sent_tokens)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

            data_x_id.append(indexed_tokens)

            # scores_and_positions = get_score_vector_positions()[prompt_id]

    return data_x_id, true_score, norm_score, prompt_ids


def get_scaled_scores(score, prompt_id_array):
    asap_score_ranges = {
        0: (0, 60),
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60)
    }
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_score_ranges[prompt_id_array]
        score = (score - low) / (high - low)
    return score


def get_data(paths, prompt_id, tokenizer):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    logger.info("Prompt id is %s" % prompt_id)

    # type_ids
    train_x_id, train_true_score, train_norm_score, train_prompt_ids, train_max_num = \
        read_dataset(train_path, prompt_id, tokenizer=tokenizer)
    dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_max_num = read_dataset(dev_path, prompt_id,
                                                                                         tokenizer=tokenizer)
    test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_max_num = read_dataset(test_path, prompt_id,
                                                                                              tokenizer=tokenizer)

    return (train_x_id, train_true_score, train_norm_score, train_prompt_ids, train_max_num), \
           (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_max_num), \
           (test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_max_num)


def get_data_with_prompt(paths, prompt_path, prompt_id, tokenizer):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    logger.info("Prompt id is %s" % prompt_id)
    prompt_list = pd.read_csv(prompt_path)
    content = prompt_list['prompt'][prompt_id - 1]
    prompt_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
    tokenized_prompt = tokenizer.tokenize(prompt_tokens)
    prompt_ids = tokenizer.convert_tokens_to_ids(tokenized_prompt)
    # type_ids
    train_x_id, train_true_score, train_norm_score, train_prompt_ids, train_max_num = \
        read_dataset(train_path, prompt_id, tokenizer=tokenizer)
    dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_max_num = read_dataset(dev_path, prompt_id,
                                                                                         tokenizer=tokenizer)
    test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_max_num = read_dataset(test_path, prompt_id,
                                                                                              tokenizer=tokenizer)

    return (train_x_id, train_true_score, train_norm_score, train_prompt_ids, train_max_num), \
           (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_max_num), \
           (test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_max_num), \
           prompt_ids


def get_data_single(paths, prompt_id, tokenizer, attribute='score'):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    logger.info("Prompt id is %s" % prompt_id)

    # type_ids
    train_x_id, train_norm_score, train_prompt_ids, train_max_num = \
        read_dataset_single(train_path, prompt_id, tokenizer=tokenizer)
    dev_x_id, dev_norm_score, dev_prompt_ids, dev_max_num = read_dataset_single(dev_path, prompt_id,
                                                                                tokenizer=tokenizer)
    test_x_id, test_norm_score, test_prompt_ids, test_max_num = read_dataset_single(test_path,
                                                                                    prompt_id,
                                                                                    tokenizer=tokenizer)

    return (train_x_id, train_norm_score, train_prompt_ids, train_max_num), \
           (dev_x_id, dev_norm_score, dev_prompt_ids, dev_max_num), \
           (test_x_id, test_norm_score, test_prompt_ids, test_max_num)


def get_data_for_ASAP(paths, prompt_id, tokenizer):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    logger.info("Prompt id is %s" % prompt_id)

    # type_ids
    train_x_id, train_true_score, train_norm_score, train_prompt_ids = \
        read_dataset_for_ASAP(prompt_id, train_path, tokenizer=tokenizer)
    dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids = \
        read_dataset_for_ASAP(prompt_id, dev_path, tokenizer=tokenizer)
    test_x_id, test_true_score, test_norm_score, test_prompt_ids = \
        read_dataset_for_ASAP(prompt_id, test_path, tokenizer=tokenizer)

    return (train_x_id, train_true_score, train_norm_score, train_prompt_ids), \
           (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids), \
           (test_x_id, test_true_score, test_norm_score, test_prompt_ids)

# -*- coding: utf-8 -*-

import utils
import numpy as np
import torch
import reader
import pickle
import pandas as pd
from sklearn import preprocessing

logger = utils.get_logger("Prepare data ...")


def get_readability_features(readability_path):
    with open(readability_path, 'rb') as fp:
        readability_features = pickle.load(fp)
    return readability_features


def get_linguistic_features(linguistic_features_path):
    features_df = pd.read_csv(linguistic_features_path)
    return features_df


def get_normalized_features(features_df):
    column_names_not_to_normalize = ['item_id', 'prompt_id', 'score']
    column_names_to_normalize = list(features_df.columns.values)
    for col in column_names_not_to_normalize:
        column_names_to_normalize.remove(col)
    final_columns = ['item_id'] + column_names_to_normalize
    normalized_features_df = None
    for prompt_ in range(1, 9):
        is_prompt_id = features_df['prompt_id'] == prompt_
        prompt_id_df = features_df[is_prompt_id]
        x = prompt_id_df[column_names_to_normalize].values
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_pd1 = min_max_scaler.fit_transform(x)
        df_temp = pd.DataFrame(normalized_pd1, columns=column_names_to_normalize, index=prompt_id_df.index)
        prompt_id_df[column_names_to_normalize] = df_temp
        final_df = prompt_id_df[final_columns]
        if normalized_features_df is not None:
            normalized_features_df = pd.concat([normalized_features_df, final_df], ignore_index=True)
        else:
            normalized_features_df = final_df
    return normalized_features_df


def prepare_sentence_data(datapaths, prompt_id=1, max_num=512, tokenizer=None):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x_id, train_true_score, train_norm_score, train_prompt_ids, train_max_num), \
    (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_max_num), \
    (test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_max_num) = reader.get_data(datapaths, prompt_id,
                                                                                                   tokenizer)

    X_train_ids, X_train_mask, Y_train = utils.padding_sentence_sequences(train_x_id, train_norm_score)
    X_dev_ids, X_dev_mask, Y_dev = utils.padding_sentence_sequences(dev_x_id, dev_norm_score)
    X_test_ids, X_test_mask, Y_test = utils.padding_sentence_sequences(test_x_id, test_norm_score)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train_ids.shape))
    logger.info('  dev X shape:   ' + str(X_dev_ids.shape))
    logger.info('  test X shape:  ' + str(X_test_ids.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    return (X_train_ids, X_train_mask, Y_train), \
           (X_dev_ids, X_dev_mask, Y_dev), \
           (X_test_ids, X_test_mask, Y_test)


def prepare_sentence_data_with_prompt(datapaths, prompt_info_path, prompt_id=1, max_num=512, tokenizer=None):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x_id, train_true_score, train_norm_score, train_prompt_ids, train_max_num), \
    (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_max_num), \
    (test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_max_num), \
    prompt_ids = reader.get_data_with_prompt(datapaths, prompt_info_path, prompt_id, tokenizer)

    X_train_ids, X_train_mask, Y_train = utils.padding_sentence_sequences(train_x_id, train_norm_score)
    X_dev_ids, X_dev_mask, Y_dev = utils.padding_sentence_sequences(dev_x_id, dev_norm_score)
    X_test_ids, X_test_mask, Y_test = utils.padding_sentence_sequences(test_x_id, test_norm_score)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train_ids.shape))
    logger.info('  dev X shape:   ' + str(X_dev_ids.shape))
    logger.info('  test X shape:  ' + str(X_test_ids.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    return (X_train_ids, X_train_mask, Y_train), \
           (X_dev_ids, X_dev_mask, Y_dev), \
           (X_test_ids, X_test_mask, Y_test), prompt_ids


def prepare_sentence_data_for_gen(datapaths, prompt_id=1, max_num=512, tokenizer=None):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x_id, train_y_id, train_prompt_ids, train_norm_score, train_label_text), \
    (dev_x_id, dev_y_id, dev_prompt_ids, dev_norm_score, dev_label_text), \
    (test_x_id, test_y_id, test_prompt_ids, test_norm_score, test_label_text) \
        = reader.get_data_for_gen(datapaths,
                                  prompt_id,
                                  tokenizer)

    X_train_ids, X_train_mask, train_scores, Y_train_ids, Y_train_mask = utils.padding_sentence_sequences_for_gen(
        train_x_id, train_norm_score,
        train_y_id)
    X_dev_ids, X_dev_mask, dev_scores, Y_dev_ids, Y_dev_mask = utils.padding_sentence_sequences_for_gen(dev_x_id,
                                                                                                        dev_norm_score,
                                                                                                        dev_y_id)
    X_test_ids, X_test_mask, test_scores, Y_test_ids, Y_test_mask = utils.padding_sentence_sequences_for_gen(test_x_id,
                                                                                                             test_norm_score,
                                                                                                             test_y_id)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train_ids.shape))
    logger.info('  dev X shape:   ' + str(X_dev_ids.shape))
    logger.info('  test X shape:  ' + str(X_test_ids.shape))

    logger.info('  train Y shape: ' + str(Y_train_ids.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev_ids.shape))
    logger.info('  test Y shape:  ' + str(Y_test_ids.shape))

    return (X_train_ids, X_train_mask, train_scores, Y_train_ids, Y_train_mask), \
           (X_dev_ids, X_dev_mask, dev_scores, Y_dev_ids, Y_dev_mask), \
           (X_test_ids, X_test_mask, test_scores, Y_test_ids, Y_test_mask)


def prepare_sentence_data_single(datapaths, prompt_id=1, max_num=512, tokenizer=None, attribute='score'):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x_id, train_norm_score, train_prompt_ids, train_max_num), \
    (dev_x_id, dev_norm_score, dev_prompt_ids, dev_max_num), \
    (test_x_id, test_norm_score, test_prompt_ids, test_max_num) = reader.get_data_single(datapaths,
                                                                                         prompt_id,
                                                                                         tokenizer, attribute)

    X_train_ids, X_train_mask, Y_train = utils.padding_sentence_sequences_single(train_x_id, train_norm_score)
    X_dev_ids, X_dev_mask, Y_dev = utils.padding_sentence_sequences_single(dev_x_id, dev_norm_score)
    X_test_ids, X_test_mask, Y_test = utils.padding_sentence_sequences_single(test_x_id, test_norm_score)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train_ids.shape))
    logger.info('  dev X shape:   ' + str(X_dev_ids.shape))
    logger.info('  test X shape:  ' + str(X_test_ids.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    return (X_train_ids, X_train_mask, Y_train), \
           (X_dev_ids, X_dev_mask, Y_dev), \
           (X_test_ids, X_test_mask, Y_test)


def prepare_sentence_data_with_fea(datapaths, features_path, prompt_id=1, tokenizer=None, essay_list=None):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"

    linguistic_features = get_linguistic_features(features_path)
    normalized_linguistic_features = get_normalized_features(linguistic_features)

    (train_x_id, train_true_score, train_norm_score, train_prompt_ids, train_max_num,
     train_features_x), \
    (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_max_num, dev_features_x), \
    (test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_max_num,
     test_features_x) = \
        reader.get_data_with_feature(datapaths, prompt_id, normalized_linguistic_features,
                                     tokenizer, essay_list=essay_list)

    X_train_ids, X_train_mask, Y_train = utils.padding_sentence_sequences(train_x_id, train_norm_score)
    X_dev_ids, X_dev_mask, Y_dev = utils.padding_sentence_sequences(dev_x_id, dev_norm_score)
    X_test_ids, X_test_mask, Y_test = utils.padding_sentence_sequences(test_x_id, test_norm_score)

    train_features_x = np.array(train_features_x, dtype=np.float32)
    dev_features_x = np.array(dev_features_x, dtype=np.float32)
    test_features_x = np.array(test_features_x, dtype=np.float32)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train_ids.shape))
    logger.info('  dev X shape:   ' + str(X_dev_ids.shape))
    logger.info('  test X shape:  ' + str(X_test_ids.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_fx shape: ' + str(train_features_x.shape))
    logger.info('  dev_fx shape:   ' + str(dev_features_x.shape))
    logger.info('  test_fx shape:  ' + str(test_features_x.shape))

    return (X_train_ids, X_train_mask, Y_train, train_features_x), \
           (X_dev_ids, X_dev_mask, Y_dev, dev_features_x), \
           (X_test_ids, X_test_mask, Y_test, test_features_x)


def prepare_sentence_data_with_LLM(datapaths, LLM_path, prompt_id=1, tokenizer=None, essay_list=None):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"

    (train_x_id, train_true_score, train_norm_score, train_prompt_ids,
     train_LLM_x), \
    (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_LLM_x), \
    (test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_LLM_x), llm_len = \
        reader.get_data_with_LLM(datapaths, prompt_id, LLM_path,
                                 tokenizer, essay_list=essay_list)

    X_train_ids, X_train_mask, Y_train = utils.padding_sentence_sequences(train_x_id, train_norm_score)
    X_dev_ids, X_dev_mask, Y_dev = utils.padding_sentence_sequences(dev_x_id, dev_norm_score)
    X_test_ids, X_test_mask, Y_test = utils.padding_sentence_sequences(test_x_id, test_norm_score)

    llm_train_ids, llm_train_mask = utils.padding_sentence_sequences_forLLM(train_LLM_x)
    llm_dev_ids, llm_dev_mask = utils.padding_sentence_sequences_forLLM(dev_LLM_x)
    llm_test_ids, llm_test_mask = utils.padding_sentence_sequences_forLLM(test_LLM_x)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train_ids.shape))
    logger.info('  dev X shape:   ' + str(X_dev_ids.shape))
    logger.info('  test X shape:  ' + str(X_test_ids.shape))

    logger.info('  train llm shape: ' + str(llm_train_ids.shape))
    logger.info('  dev llm shape:   ' + str(llm_dev_ids.shape))
    logger.info('  test llm shape:  ' + str(llm_test_ids.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    return (X_train_ids, X_train_mask, Y_train, llm_train_ids, llm_train_mask), \
           (X_dev_ids, X_dev_mask, Y_dev, llm_dev_ids, llm_dev_mask), \
           (X_test_ids, X_test_mask, Y_test, llm_test_ids, llm_test_mask)


def prepare_sentence_data_for_ASAP(datapaths, prompt_id=1, tokenizer=None):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"

    (train_x_id, train_true_score, train_norm_score, train_prompt_ids), \
    (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids), \
    (test_x_id, test_true_score, test_norm_score, test_prompt_ids) = \
        reader.get_data_for_ASAP(datapaths, prompt_id, tokenizer)

    X_train_ids, X_train_mask, Y_train = utils.padding_sentence_sequences(train_x_id, train_norm_score)
    X_dev_ids, X_dev_mask, Y_dev = utils.padding_sentence_sequences(dev_x_id, dev_norm_score)
    X_test_ids, X_test_mask, Y_test = utils.padding_sentence_sequences(test_x_id, test_norm_score)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train_ids.shape))
    logger.info('  dev X shape:   ' + str(X_dev_ids.shape))
    logger.info('  test X shape:  ' + str(X_test_ids.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    return (X_train_ids, X_train_mask, Y_train), \
           (X_dev_ids, X_dev_mask, Y_dev), \
           (X_test_ids, X_test_mask, Y_test)


def prepare_sentence_data_with_LLMs(datapaths, LLM_path, prompt_id=1, tokenizer=None, essay_list=None, llm_index=-1):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"

    (train_x_id, train_true_score, train_norm_score, train_prompt_ids, train_LLM_x), \
    (dev_x_id, dev_true_score, dev_norm_score, dev_prompt_ids, dev_LLM_x), \
    (test_x_id, test_true_score, test_norm_score, test_prompt_ids, test_LLM_x) = \
        reader.get_data_with_LLMs(datapaths, prompt_id, LLM_path,
                                  tokenizer, essay_list=essay_list, llm_index=llm_index)

    X_train_ids, X_train_mask, Y_train = utils.padding_sentence_sequences(train_x_id, train_norm_score)
    X_dev_ids, X_dev_mask, Y_dev = utils.padding_sentence_sequences(dev_x_id, dev_norm_score)
    X_test_ids, X_test_mask, Y_test = utils.padding_sentence_sequences(test_x_id, test_norm_score)

    llm_train_ids, llm_train_mask, _ = utils.padding_sentence_sequences(train_LLM_x, train_norm_score)
    llm_dev_ids, llm_dev_mask, _ = utils.padding_sentence_sequences(dev_LLM_x, dev_norm_score)
    llm_test_ids, llm_test_mask, _ = utils.padding_sentence_sequences(test_LLM_x, test_norm_score)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train_ids.shape))
    logger.info('  dev X shape:   ' + str(X_dev_ids.shape))
    logger.info('  test X shape:  ' + str(X_test_ids.shape))

    logger.info('  train llm shape: ' + str(llm_train_ids.shape))
    logger.info('  dev llm shape:   ' + str(llm_dev_ids.shape))
    logger.info('  test llm shape:  ' + str(llm_test_ids.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    return (X_train_ids, X_train_mask, Y_train, llm_train_ids, llm_train_mask), \
           (X_dev_ids, X_dev_mask, Y_dev, llm_dev_ids, llm_dev_mask), \
           (X_test_ids, X_test_mask, Y_test, llm_test_ids, llm_test_mask)


def data_shuffle(features):
    queries_test = []
    for i in range(len(features)):
        queries = np.random.choice(len(features[i][0]), len(features[i][0]), replace=False)
        queries_x = []
        queries_y = []
        for loc in queries:
            queries_x.append(features[i][0][loc])
            queries_y.append(features[i][1][loc])
        queries_y = np.squeeze(np.broadcast_arrays(queries_y))
        queries_test.append((queries_x, queries_y))

    return queries_test


# NPCR
def data_pre_opti(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, prompt_id, example_size):
    features_train = []
    y_train = []

    # C2 performs better than C1
    for i in range(X_train.shape[0] - 6):
        # C2
        if Y_train[i] != Y_train[i + 1]:
            y_train.append(Y_train[i] - Y_train[i + 1])
            features_train.append((X_train[i], X_train[i + 1]))
        if Y_train[i] != Y_train[i + 2]:
            y_train.append(Y_train[i] - Y_train[i + 2])
            features_train.append((X_train[i], X_train[i + 2]))
        if Y_train[i] != Y_train[i + 3]:
            y_train.append(Y_train[i] - Y_train[i + 3])
            features_train.append((X_train[i], X_train[i + 3]))
    features_train = np.array(features_train)
    y_train = np.array(y_train)
    y_train = reader.get_model_friendly_scores(y_train, prompt_id)

    train_x0 = [j[0] for j in features_train]
    train_x1 = [j[1] for j in features_train]

    train_x0 = torch.LongTensor(train_x0)
    train_x1 = torch.LongTensor(train_x1)

    # dev
    features_dev = []
    dev_y_example = []
    dev_y_goal = []

    maxTrainLen = X_train.shape[0] - 1
    maxR = X_train.shape[0] - 1
    minL = 0
    for i in range(X_dev.shape[0]):
        num = example_size
        j = 0
        while num > 0:
            if Y_train[j] != Y_dev[i]:
                num -= 1
                features_dev.append((X_dev[i], X_train[j]))
                dev_y_example.append(Y_train[j])
            j += 1

        dev_y_goal.append(Y_dev[i])
    features_dev = np.array(features_dev)
    dev_y_example = np.array(dev_y_example)
    dev_y_goal = np.array(dev_y_goal)

    # test
    features_test = []
    test_y_example = []
    test_y_goal = []

    maxR = X_train.shape[0] - 1
    minL = 0
    for i in range(X_test.shape[0]):
        num = example_size
        j = 0
        while num > 0:
            if Y_train[j] != Y_test[i]:
                num -= 1
                features_test.append((X_test[i], X_train[j]))
                test_y_example.append(Y_train[j])
            j += 1

        test_y_goal.append(Y_test[i])
    features_test = np.array(features_test)
    test_y_example = np.array(test_y_example)
    test_y_goal = np.array(test_y_goal)

    return train_x0, train_x1, y_train, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal

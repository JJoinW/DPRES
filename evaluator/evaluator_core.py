from utils import get_logger
from .metrics import *
import numpy as np
import torch.utils.data as Data
import torch
import os

cur_dir = os.getcwd()
ckpt_dir = 'checkpoints'
dir = os.path.join(cur_dir, ckpt_dir)
os.makedirs(dir, exist_ok=True)

logger = get_logger("Evaluate stats")


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


def get_score_vector_positions():
    return {
        1: {'score': 0, 'content': 1, 'organization': 2, 'word_choice': 3,
            'sentence_fluency': 4, 'conventions': 5},
        2: {'score': 0, 'content': 1, 'organization': 2, 'word_choice': 3,
            'sentence_fluency': 4, 'conventions': 5},
        3: {'score': 0, 'content': 1, 'prompt_adherence': 2, 'language': 3, 'narrativity': 4},
        4: {'score': 0, 'content': 1, 'prompt_adherence': 2, 'language': 3, 'narrativity': 4},
        5: {'score': 0, 'content': 1, 'prompt_adherence': 2, 'language': 3, 'narrativity': 4},
        6: {'score': 0, 'content': 1, 'prompt_adherence': 2, 'language': 3, 'narrativity': 4},
        7: {'score': 0, 'content': 1, 'organization': 2, 'conventions': 3, 'style': 4},
        8: {'score': 0, 'content': 1, 'organization': 2, 'word_choice': 3, 'sentence_fluency': 4,
            'conventions': 5, 'voice': 6}}


def get_min_max_scores():
    return {
        1: {'score': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'score': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6), 'style': (0, 6)},
        8: {'score': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
            'sentence_fluency': (2, 12), 'conventions': (2, 12), 'voice': (2, 12)}}


def separate_and_rescale_attributes_for_scoring(scores, set_id):
    score_vector_positions = get_score_vector_positions()[set_id]
    min_max_scores = get_min_max_scores()[set_id]
    individual_att_scores_dict = {}
    for att_scores in scores:
        for relevant_attribute in min_max_scores.keys():
            min_score = min_max_scores[relevant_attribute][0]
            max_score = min_max_scores[relevant_attribute][1]
            att_position = score_vector_positions[relevant_attribute]
            att_score = att_scores[att_position]
            rescaled_score = att_score * (max_score - min_score) + min_score
            try:
                individual_att_scores_dict[relevant_attribute].append(np.around(rescaled_score).astype(int))
            except KeyError:
                individual_att_scores_dict[relevant_attribute] = [np.around(rescaled_score).astype(int)]
    return individual_att_scores_dict


def separate_and_rescale_attributes_for_scoring_feedback(scores):
    score_vector_positions = {'cohesion': 0, 'syntax': 1, 'vocabulary': 2, 'phraseology': 3, 'grammar': 4,
                              'conventions': 5}
    min_max_scores = {'cohesion': (1, 5), 'syntax': (1, 5), 'vocabulary': (1, 5), 'phraseology': (1, 5),
                      'grammar': (1, 5), 'conventions': (1, 5)}
    individual_att_scores_dict = {}
    for att_scores in scores:
        for relevant_attribute in min_max_scores.keys():
            min_score = min_max_scores[relevant_attribute][0]
            max_score = min_max_scores[relevant_attribute][1]
            att_position = score_vector_positions[relevant_attribute]
            att_score = att_scores[att_position]
            rescaled_score = att_score * (max_score - min_score) + min_score
            try:
                individual_att_scores_dict[relevant_attribute].append(np.around(rescaled_score).astype(int))
            except KeyError:
                individual_att_scores_dict[relevant_attribute] = [np.around(rescaled_score).astype(int)]
    return individual_att_scores_dict


class Evaluator_normal:
    def __init__(self, prompt_id, dev_x, dev_y, test_x, test_y, device):
        self.best_dev_epoch = -1
        self.best_test_kappa_set = None
        self.best_dev_kappa_set = -1.0
        self.best_test_kappa_mean = -1.0
        self.best_dev_kappa_mean = -1.0
        self.dev_kappa_mean = 0.0
        self.test_kappa_mean = 0.0
        self.prompt_id = prompt_id
        self.dev_x, self.test_x = dev_x, test_x
        self.dev_y, self.test_y = dev_y, test_y
        self.device = device
        self.best_dev = [-1, -1, -1, -1]
        self.dev_test = [-1, -1, -1, -1]
        self.best_test = [-1, -1, -1, -1]
        self.test_dev = [-1, -1, -1, -1]

        self.dev_loader, self.test_loader = self.init_data_loader()

    def calc_correl(self, dev_true, test_true, dev_pred, test_pred):
        self.dev_pr = pearson(dev_true, dev_pred)
        self.test_pr = pearson(test_true, test_pred)

        self.dev_spr = spearman(dev_true, dev_pred)
        self.test_spr = spearman(test_true, test_pred)

    def calc_kappa(self, pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    def calc_rmse(self, dev_true, test_true, dev_pred, test_pred):
        self.dev_rmse = root_mean_square_error(dev_true, dev_pred)
        self.test_rmse = root_mean_square_error(test_true, test_pred)

    def init_data_loader(self):
        dev_x0 = [j for j in self.dev_x[0]]
        dev_x1 = [j for j in self.dev_x[-1]]
        dev_x0 = torch.LongTensor(dev_x0)
        dev_x1 = torch.LongTensor(dev_x1)
        dev_y = torch.Tensor(self.dev_y)

        test_x0 = [j for j in self.test_x[0]]
        test_x1 = [j for j in self.test_x[-1]]
        test_x0 = torch.LongTensor(test_x0)
        test_x1 = torch.LongTensor(test_x1)
        test_y = torch.Tensor(self.test_y)

        dev_dataset = Data.TensorDataset(dev_x0, dev_x1, dev_y)
        test_dataset = Data.TensorDataset(test_x0, test_x1, test_y)

        dev_loader = Data.DataLoader(
            dataset=dev_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=dev_dataset), batch_size=32, drop_last=False
            ),
            num_workers=2
        )
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=test_dataset), batch_size=32, drop_last=False
            ),
            num_workers=2
        )

        return dev_loader, test_loader

    def evaluate(self, model, epoch, print_info=False, res_log=None):
        dev_pred_int = []
        dev_true_int = []
        for step, (batch_dev_x0, batch_dev_x1, batch_example_s) in enumerate(self.dev_loader):
            batch_true = batch_example_s.tolist()
            dev_true_int.extend(batch_true)

            dev_y_pred = model(batch_dev_x0.to(self.device), batch_dev_x1.to(self.device))
            dev_y_pred = dev_y_pred.cpu()
            if dev_y_pred.shape[0] > 1:
                dev_pred_i = dev_y_pred.detach().numpy().squeeze()
            else:
                dev_pred_i = dev_y_pred.detach().numpy()
            dev_pred_int.extend(dev_pred_i)

        dev_true_dict = separate_and_rescale_attributes_for_scoring(dev_true_int, self.prompt_id)
        dev_pred_dict = separate_and_rescale_attributes_for_scoring(dev_pred_int, self.prompt_id)

        test_pred_int = []
        test_true_int = []
        for step, (batch_test_x0, batch_test_x1, batch_example_s) in enumerate(self.test_loader):
            batch_true = batch_example_s.tolist()
            test_true_int.extend(batch_true)

            test_y_pred = model(batch_test_x0.to(self.device), batch_test_x1.to(self.device))
            test_y_pred = test_y_pred.cpu()
            if test_y_pred.shape[0] > 1:
                test_pred_i = test_y_pred.detach().numpy().squeeze()
            else:
                test_pred_i = test_y_pred.detach().numpy().squeeze(axis=2)
            test_pred_int.extend(test_pred_i)
        test_true_dict = separate_and_rescale_attributes_for_scoring(test_true_int, self.prompt_id)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred_int, self.prompt_id)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], dev_true_dict[key]) for key in
                          dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], test_true_dict[key]) for key in
                           test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            # file_path = os.path.join(dir, "checkpoint_best" + str(self.prompt_id) + ".ckpt")
            # model.save_weights(file_path)  # save the best model
            # print("Save best model to ", file_path)
        if print_info:
            self.print_info(epoch, res_log)

    def print_info(self, epoch, res_log):
        print('CURRENT EPOCH: {}'.format(epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        res_log.write('CURRENT EPOCH: {} \n'.format(epoch))
        res_log.write('[DEV] AVG QWK: {} \n'.format(round(self.dev_kappa_mean, 3)))
        res_log.flush()
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(att, round(self.kappa_dev[att], 3)))
            res_log.write('[DEV] {} QWK: {} \n'.format(att, round(self.kappa_dev[att], 3)))
            res_log.flush()
        res_log.write('------------------------ \n')
        res_log.flush()
        print(
            '------------------------')
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        res_log.write('[TEST] AVG QWK: {} \n'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(att, round(self.kappa_test[att], 3)))
            res_log.write('[TEST] {} QWK: {} \n'.format(att, round(self.kappa_test[att], 3)))
            res_log.flush()

        print(
            '------------------------')
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        res_log.write(
            '[BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))

        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
            res_log.write('[BEST TEST] {} QWK: {} \n'.format(att, round(self.best_test_kappa_set[att], 3)))
            res_log.flush()
        res_log.write('------------------------ \n')
        res_log.flush()
        print(
            '--------------------------------------------------------------------------------------------------'
            '------------------------')

    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print(
            '---------------------------------------------------------------------------------------------------'
            '-----------------------')


class Evaluator_normal_feature:
    def __init__(self, prompt_id, dev_x, dev_y, test_x, test_y, device):
        self.best_dev_epoch = -1
        self.best_test_kappa_set = None
        self.best_dev_kappa_set = -1.0
        self.best_test_kappa_mean = -1.0
        self.best_dev_kappa_mean = -1.0
        self.dev_kappa_mean = 0.0
        self.test_kappa_mean = 0.0
        self.test_best_value = -1.0
        self.best_test_epoch = -1
        self.prompt_id = prompt_id
        self.dev_x, self.test_x = dev_x, test_x
        self.dev_y, self.test_y = dev_y, test_y
        self.device = device
        self.best_dev = [-1, -1, -1, -1]
        self.dev_test = [-1, -1, -1, -1]
        self.best_test = [-1, -1, -1, -1]
        self.test_dev = [-1, -1, -1, -1]

        self.dev_loader, self.test_loader = self.init_data_loader()

    def calc_correl(self, dev_true, test_true, dev_pred, test_pred):
        self.dev_pr = pearson(dev_true, dev_pred)
        self.test_pr = pearson(test_true, test_pred)

        self.dev_spr = spearman(dev_true, dev_pred)
        self.test_spr = spearman(test_true, test_pred)

    def calc_kappa(self, pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    def calc_rmse(self, dev_true, test_true, dev_pred, test_pred):
        self.dev_rmse = root_mean_square_error(dev_true, dev_pred)
        self.test_rmse = root_mean_square_error(test_true, test_pred)

    def init_data_loader(self):
        dev_x0 = [j for j in self.dev_x[0]]
        dev_x1 = [j for j in self.dev_x[1]]
        dev_x0 = torch.LongTensor(dev_x0)
        dev_x1 = torch.LongTensor(dev_x1)
        dev_y = torch.Tensor(self.dev_y)

        test_x0 = [j for j in self.test_x[0]]
        test_x1 = [j for j in self.test_x[1]]
        test_x0 = torch.LongTensor(test_x0)
        test_x1 = torch.LongTensor(test_x1)
        test_y = torch.Tensor(self.test_y)

        dev_dataset = Data.TensorDataset(dev_x0, dev_x1, dev_y)
        test_dataset = Data.TensorDataset(test_x0, test_x1, test_y)

        dev_loader = Data.DataLoader(
            dataset=dev_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=dev_dataset), batch_size=32, drop_last=False
            ),
            num_workers=2
        )
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=test_dataset), batch_size=32, drop_last=False
            ),
            num_workers=2
        )

        return dev_loader, test_loader

    def evaluate(self, model, epoch, print_info=False, res_log=None, model_name=''):
        dev_pred_int = []
        dev_true_int = []
        trait_num = len(get_score_vector_positions()[self.prompt_id])
        for step, (batch_dev_x0, batch_dev_x1, batch_example_s) in enumerate(
                self.dev_loader):
            batch_true = batch_example_s.tolist()
            dev_true_int.extend(batch_true)

            dev_y_pred = model(batch_dev_x0.to(self.device), batch_dev_x1.to(self.device))[0]
            dev_y_pred = dev_y_pred.cpu().detach().numpy().reshape(-1, trait_num)
            if dev_y_pred.shape[0] > 1:
                dev_pred_i = dev_y_pred.squeeze()
            else:
                dev_pred_i = np.reshape(dev_y_pred, (1, -1))
            dev_pred_int.extend(dev_pred_i)

        dev_true_dict = separate_and_rescale_attributes_for_scoring(dev_true_int, self.prompt_id)
        dev_pred_dict = separate_and_rescale_attributes_for_scoring(dev_pred_int, self.prompt_id)

        test_pred_int = []
        test_true_int = []
        for step, (batch_test_x0, batch_test_x1, batch_example_s) in enumerate(
                self.test_loader):
            batch_true = batch_example_s.tolist()
            test_true_int.extend(batch_true)

            test_y_pred = model(batch_test_x0.to(self.device), batch_test_x1.to(self.device))[0]
            test_y_pred = test_y_pred.cpu().detach().numpy().reshape(-1, trait_num)
            if test_y_pred.shape[0] > 1:
                test_pred_i = test_y_pred.squeeze()
            else:
                test_pred_i = np.reshape(test_y_pred, (1, -1))
            test_pred_int.extend(test_pred_i)
        test_true_dict = separate_and_rescale_attributes_for_scoring(test_true_int, self.prompt_id)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred_int, self.prompt_id)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], dev_true_dict[key]) for key in
                          dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], test_true_dict[key]) for key in
                           test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            file_path = os.path.join(dir, f"{model_name}-checkpoint_best-" + str(self.prompt_id) + ".ckpt")
            torch.save(model, file_path)  # save the best model
            print("Save best model to ", file_path)
        if self.test_kappa_mean > self.test_best_value:
            self.test_best_value = self.test_kappa_mean
            self.best_test_epoch = epoch
        if print_info:
            self.print_info(epoch, res_log)

    def evaluate_base(self, model, epoch, print_info=False, res_log=None, model_name=''):
        dev_pred_int = []
        dev_true_int = []
        for step, (batch_dev_x0, batch_dev_x1, batch_dev_f, batch_example_s) in enumerate(self.dev_loader):
            batch_true = batch_example_s.tolist()
            dev_true_int.extend(batch_true)

            dev_y_pred = model(batch_dev_x0.to(self.device), batch_dev_x1.to(self.device),
                               batch_dev_f.to(self.device))
            dev_y_pred = dev_y_pred.cpu()
            if dev_y_pred.shape[0] > 1:
                dev_pred_i = dev_y_pred.detach().numpy().squeeze()
            else:
                dev_pred_i = dev_y_pred.detach().numpy()
            dev_pred_int.extend(dev_pred_i)

        dev_true_dict = separate_and_rescale_attributes_for_scoring(dev_true_int, self.prompt_id)
        dev_pred_dict = separate_and_rescale_attributes_for_scoring(dev_pred_int, self.prompt_id)

        test_pred_int = []
        test_true_int = []
        for step, (batch_test_x0, batch_test_x1, batch_test_f, batch_example_s) in enumerate(self.test_loader):
            batch_true = batch_example_s.tolist()
            test_true_int.extend(batch_true)

            test_y_pred = model(batch_test_x0.to(self.device), batch_test_x1.to(self.device),
                                batch_test_f.to(self.device))
            test_y_pred = test_y_pred.cpu()
            if test_y_pred.shape[0] > 1:
                test_pred_i = test_y_pred.detach().numpy().squeeze()
            else:
                test_pred_i = test_y_pred.detach().numpy().squeeze(axis=2)
            test_pred_int.extend(test_pred_i)
        test_true_dict = separate_and_rescale_attributes_for_scoring(test_true_int, self.prompt_id)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred_int, self.prompt_id)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], dev_true_dict[key]) for key in
                          dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], test_true_dict[key]) for key in
                           test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            file_path = os.path.join(dir, f"{model_name}-checkpoint_best-" + str(self.prompt_id) + ".ckpt")
            torch.save(model, file_path)  # save the best model
            print("Save best model to ", file_path)
        if self.test_kappa_mean > self.test_best_value:
            self.test_best_value = self.test_kappa_mean
            self.best_test_epoch = epoch
        if print_info:
            self.print_info(epoch, res_log)

    def print_info(self, epoch, res_log):
        print('CURRENT EPOCH: {}'.format(epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        res_log.write('CURRENT EPOCH: {} \n'.format(epoch))
        res_log.write('[DEV] AVG QWK: {} \n'.format(round(self.dev_kappa_mean, 3)))
        res_log.flush()
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(att, round(self.kappa_dev[att], 3)))
            res_log.write('[DEV] {} QWK: {} \n'.format(att, round(self.kappa_dev[att], 3)))
            res_log.flush()
        res_log.write('------------------------ \n')
        res_log.flush()
        print(
            '------------------------')
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        res_log.write('[TEST] AVG QWK: {} \n'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(att, round(self.kappa_test[att], 3)))
            res_log.write('[TEST] {} QWK: {} \n'.format(att, round(self.kappa_test[att], 3)))
            res_log.flush()

        print(
            '------------------------')
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        res_log.write(
            '[BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))

        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
            res_log.write('[BEST TEST] {} QWK: {} \n'.format(att, round(self.best_test_kappa_set[att], 3)))
            res_log.flush()
        res_log.write('------------------------ \n')
        print('[THE BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.test_best_value, 3), self.best_test_epoch))
        res_log.write(
            '[THE BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.test_best_value, 3), self.best_test_epoch))
        res_log.flush()
        print(
            '--------------------------------------------------------------------------------------------------'
            '------------------------')

    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print(
            '---------------------------------------------------------------------------------------------------'
            '-----------------------')


class Evaluator_normal_feature_LLM:
    def __init__(self, prompt_id, dev_x, dev_y, test_x, test_y, device):
        self.best_dev_epoch = -1
        self.best_test_kappa_set = None
        self.best_dev_kappa_set = -1.0
        self.best_test_kappa_mean = -1.0
        self.best_dev_kappa_mean = -1.0
        self.dev_kappa_mean = 0.0
        self.test_kappa_mean = 0.0
        self.test_best_value = -1.0
        self.best_test_epoch = -1
        self.prompt_id = prompt_id
        self.dev_x, self.test_x = dev_x, test_x
        self.dev_y, self.test_y = dev_y, test_y
        self.device = device
        self.best_dev = [-1, -1, -1, -1]
        self.dev_test = [-1, -1, -1, -1]
        self.best_test = [-1, -1, -1, -1]
        self.test_dev = [-1, -1, -1, -1]

        self.dev_loader, self.test_loader = self.init_data_loader()

    def calc_correl(self, dev_true, test_true, dev_pred, test_pred):
        self.dev_pr = pearson(dev_true, dev_pred)
        self.test_pr = pearson(test_true, test_pred)

        self.dev_spr = spearman(dev_true, dev_pred)
        self.test_spr = spearman(test_true, test_pred)

    def calc_kappa(self, pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    def calc_rmse(self, dev_true, test_true, dev_pred, test_pred):
        self.dev_rmse = root_mean_square_error(dev_true, dev_pred)
        self.test_rmse = root_mean_square_error(test_true, test_pred)

    def init_data_loader(self):
        dev_x0 = [j for j in self.dev_x[0]]
        dev_x1 = [j for j in self.dev_x[1]]
        dev_llm = [j for j in self.dev_x[2]]
        dev_llm_mask = [j for j in self.dev_x[3]]
        dev_x0 = torch.LongTensor(dev_x0)
        dev_x1 = torch.LongTensor(dev_x1)
        dev_llm = torch.LongTensor(dev_llm)
        dev_llm_mask = torch.LongTensor(dev_llm_mask)
        dev_y = torch.Tensor(self.dev_y)

        test_x0 = [j for j in self.test_x[0]]
        test_x1 = [j for j in self.test_x[1]]
        test_llm = [j for j in self.test_x[2]]
        test_llm_mask = [j for j in self.test_x[3]]
        test_x0 = torch.LongTensor(test_x0)
        test_x1 = torch.LongTensor(test_x1)
        test_llm = torch.LongTensor(test_llm)
        test_llm_mask = torch.LongTensor(test_llm_mask)
        test_y = torch.Tensor(self.test_y)

        dev_dataset = Data.TensorDataset(dev_x0, dev_x1, dev_llm, dev_llm_mask, dev_y)
        test_dataset = Data.TensorDataset(test_x0, test_x1, test_llm, test_llm_mask, test_y)

        dev_loader = Data.DataLoader(
            dataset=dev_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=dev_dataset), batch_size=32, drop_last=False
            ),
            num_workers=2
        )
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=test_dataset), batch_size=32, drop_last=False
            ),
            num_workers=2
        )

        return dev_loader, test_loader

    def evaluate(self, model, epoch, print_info=False, res_log=None, model_name=''):
        dev_pred_int = []
        dev_true_int = []
        trait_num = len(get_score_vector_positions()[self.prompt_id])
        for step, (batch_dev_x0, batch_dev_x1, batch_dev_llm, batch_dev_llm_mask, batch_example_s) in enumerate(
                self.dev_loader):
            batch_true = batch_example_s.tolist()
            dev_true_int.extend(batch_true)

            dev_y_pred = model(batch_dev_x0.to(self.device), batch_dev_x1.to(self.device),
                               batch_dev_llm.to(self.device), batch_dev_llm_mask.to(self.device))[0]
            dev_y_pred = dev_y_pred.cpu().detach().numpy().reshape(-1, trait_num)
            if dev_y_pred.shape[0] > 1:
                dev_pred_i = dev_y_pred.squeeze()
            else:
                dev_pred_i = np.reshape(dev_y_pred, (1, -1))
            dev_pred_int.extend(dev_pred_i)

        dev_true_dict = separate_and_rescale_attributes_for_scoring(dev_true_int, self.prompt_id)
        dev_pred_dict = separate_and_rescale_attributes_for_scoring(dev_pred_int, self.prompt_id)

        test_pred_int = []
        test_true_int = []
        for step, (batch_test_x0, batch_test_x1, batch_test_llm, batch_test_llm_mask, batch_example_s) in enumerate(
                self.test_loader):
            batch_true = batch_example_s.tolist()
            test_true_int.extend(batch_true)

            test_y_pred = model(batch_test_x0.to(self.device), batch_test_x1.to(self.device),
                                batch_test_llm.to(self.device), batch_test_llm_mask.to(self.device))[0]
            test_y_pred = test_y_pred.cpu().detach().numpy().reshape(-1, trait_num)
            if test_y_pred.shape[0] > 1:
                test_pred_i = test_y_pred.squeeze()
            else:
                test_pred_i = np.reshape(test_y_pred, (1, -1))
            test_pred_int.extend(test_pred_i)
        test_true_dict = separate_and_rescale_attributes_for_scoring(test_true_int, self.prompt_id)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred_int, self.prompt_id)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], dev_true_dict[key]) for key in
                          dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], test_true_dict[key]) for key in
                           test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))
        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            file_path = os.path.join(dir, f"{model_name}-checkpoint_best-" + str(self.prompt_id) + ".ckpt")
            torch.save(model, file_path)  # save the best model
            print("Save best model to ", file_path)
        if self.test_kappa_mean > self.test_best_value:
            self.test_best_value = self.test_kappa_mean
            self.best_test_epoch = epoch
        if print_info:
            self.print_info(epoch, res_log)

    def print_info(self, epoch, res_log):
        print('CURRENT EPOCH: {}'.format(epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        res_log.write('CURRENT EPOCH: {} \n'.format(epoch))
        res_log.write('[DEV] AVG QWK: {} \n'.format(round(self.dev_kappa_mean, 3)))
        res_log.flush()
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(att, round(self.kappa_dev[att], 3)))
            res_log.write('[DEV] {} QWK: {} \n'.format(att, round(self.kappa_dev[att], 3)))
            res_log.flush()
        res_log.write('------------------------ \n')
        res_log.flush()
        print(
            '------------------------')
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        res_log.write('[TEST] AVG QWK: {} \n'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(att, round(self.kappa_test[att], 3)))
            res_log.write('[TEST] {} QWK: {} \n'.format(att, round(self.kappa_test[att], 3)))
            res_log.flush()

        print(
            '------------------------')
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        res_log.write(
            '[BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))

        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
            res_log.write('[BEST TEST] {} QWK: {} \n'.format(att, round(self.best_test_kappa_set[att], 3)))
            res_log.flush()
        res_log.write('------------------------ \n')
        print('[THE BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.test_best_value, 3), self.best_test_epoch))
        res_log.write(
            '[THE BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.test_best_value, 3), self.best_test_epoch))
        res_log.flush()
        print(
            '--------------------------------------------------------------------------------------------------'
            '------------------------')

    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print(
            '---------------------------------------------------------------------------------------------------'
            '-----------------------')


class Evaluator_Multi:
    def __init__(self, prompt_id, dev_x, dev_y, test_x, test_y, device):
        self.best_dev_epoch = -1
        self.best_test_kappa_set = None
        self.best_dev_kappa_set = -1.0
        self.best_test_kappa_mean = -1.0
        self.best_dev_kappa_mean = -1.0
        self.dev_kappa_mean = 0.0
        self.test_kappa_mean = 0.0
        self.test_best_value = -1.0
        self.best_test_epoch = -1
        self.prompt_id = prompt_id
        self.dev_x, self.test_x = dev_x, test_x
        self.dev_y, self.test_y = dev_y, test_y
        self.device = device
        self.best_dev = [-1, -1, -1, -1]
        self.dev_test = [-1, -1, -1, -1]
        self.best_test = [-1, -1, -1, -1]
        self.test_dev = [-1, -1, -1, -1]

        self.dev_loader, self.test_loader = self.init_data_loader()

    def calc_correl(self, dev_true, test_true, dev_pred, test_pred):
        self.dev_pr = pearson(dev_true, dev_pred)
        self.test_pr = pearson(test_true, test_pred)

        self.dev_spr = spearman(dev_true, dev_pred)
        self.test_spr = spearman(test_true, test_pred)

    def calc_kappa(self, pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    def calc_rmse(self, dev_true, test_true, dev_pred, test_pred):
        self.dev_rmse = root_mean_square_error(dev_true, dev_pred)
        self.test_rmse = root_mean_square_error(test_true, test_pred)

    def init_data_loader(self):
        dev_x0 = [j for j in self.dev_x[0]]
        dev_x1 = [j for j in self.dev_x[1]]
        dev_llm = [j for j in self.dev_x[2]]
        dev_llm_mask = [j for j in self.dev_x[3]]
        dev_x0 = torch.LongTensor(dev_x0)
        dev_x1 = torch.LongTensor(dev_x1)
        dev_llm = torch.LongTensor(dev_llm)
        dev_llm_mask = torch.LongTensor(dev_llm_mask)
        dev_y = torch.Tensor(self.dev_y)

        test_x0 = [j for j in self.test_x[0]]
        test_x1 = [j for j in self.test_x[1]]
        test_llm = [j for j in self.test_x[2]]
        test_llm_mask = [j for j in self.test_x[3]]
        test_x0 = torch.LongTensor(test_x0)
        test_x1 = torch.LongTensor(test_x1)
        test_llm = torch.LongTensor(test_llm)
        test_llm_mask = torch.LongTensor(test_llm_mask)
        test_y = torch.Tensor(self.test_y)

        dev_dataset = Data.TensorDataset(dev_x0, dev_x1, dev_llm, dev_llm_mask, dev_y)
        test_dataset = Data.TensorDataset(test_x0, test_x1, test_llm, test_llm_mask, test_y)

        dev_loader = Data.DataLoader(
            dataset=dev_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=dev_dataset), batch_size=32, drop_last=False
            ),
            num_workers=2
        )
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=test_dataset), batch_size=32, drop_last=False
            ),
            num_workers=2
        )

        return dev_loader, test_loader

    def evaluate(self, model, epoch, print_info=False, res_log=None, model_name=''):
        dev_pred_int = []
        dev_true_int = []
        for step, (batch_dev_x0, batch_dev_x1, batch_example_s) in enumerate(self.dev_loader):
            batch_true = batch_example_s.tolist()
            dev_true_int.extend(batch_true)

            dev_y_pred = model(batch_dev_x0.to(self.device), batch_dev_x1.to(self.device))[0]
            dev_y_pred = dev_y_pred.cpu()
            if batch_example_s.shape[0] > 1:
                dev_pred_i = dev_y_pred.detach().numpy().squeeze()
            else:
                dev_pred_i = dev_y_pred.detach().numpy()
                dev_pred_i = np.reshape(dev_pred_i, (1, -1))
            dev_pred_int.extend(dev_pred_i)

        dev_true_dict = separate_and_rescale_attributes_for_scoring(dev_true_int, self.prompt_id)
        dev_pred_dict = separate_and_rescale_attributes_for_scoring(dev_pred_int, self.prompt_id)

        test_pred_int = []
        test_true_int = []
        for step, (batch_test_x0, batch_test_x1, batch_example_s) in enumerate(self.test_loader):
            batch_true = batch_example_s.tolist()
            test_true_int.extend(batch_true)

            test_y_pred = model(batch_test_x0.to(self.device), batch_test_x1.to(self.device))[0]
            test_y_pred = test_y_pred.cpu()
            if batch_example_s.shape[0] > 1:
                test_pred_i = test_y_pred.detach().numpy().squeeze()
            else:
                test_pred_i = test_y_pred.detach().numpy()
                test_pred_i = np.reshape(test_pred_i, (1, -1))
            test_pred_int.extend(test_pred_i)
        test_true_dict = separate_and_rescale_attributes_for_scoring(test_true_int, self.prompt_id)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred_int, self.prompt_id)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], dev_true_dict[key]) for key in
                          dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], test_true_dict[key]) for key in
                           test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            file_path = os.path.join(dir, f"{model_name}-" + str(self.prompt_id) + ".ckpt")
            torch.save(model, file_path)  # save the best model
            print("Save best model to ", file_path)
        if self.test_kappa_mean > self.test_best_value:
            self.test_best_value = self.test_kappa_mean
            self.best_test_epoch = epoch
        if print_info:
            self.print_info(epoch, res_log)

    def evaluate_feedback(self, model, epoch, print_info=False, res_log=None, model_name=''):
        dev_pred_int = []
        dev_true_int = []
        score_vector_positions = {'cohesion': 0, 'syntax': 1, 'vocabulary': 2, 'phraseology': 3, 'grammar': 4,
                                  'conventions': 5}
        trait_num = len(score_vector_positions)
        for step, (batch_dev_x0, batch_dev_x1, batch_dev_llm, batch_dev_llm_mask, batch_example_s) in enumerate(
                self.dev_loader):
            batch_true = batch_example_s.tolist()
            dev_true_int.extend(batch_true)

            dev_y_pred = model(batch_dev_x0.to(self.device), batch_dev_x1.to(self.device),
                               batch_dev_llm.to(self.device), batch_dev_llm_mask.to(self.device))[0]
            dev_y_pred = dev_y_pred.cpu().detach().numpy().reshape(-1, trait_num)
            if dev_y_pred.shape[0] > 1:
                dev_pred_i = dev_y_pred.squeeze()
            else:
                dev_pred_i = np.reshape(dev_y_pred, (1, -1))
            dev_pred_int.extend(dev_pred_i)

        dev_true_dict = separate_and_rescale_attributes_for_scoring_feedback(dev_true_int)
        dev_pred_dict = separate_and_rescale_attributes_for_scoring_feedback(dev_pred_int)

        test_pred_int = []
        test_true_int = []
        for step, (batch_test_x0, batch_test_x1, batch_test_llm, batch_test_llm_mask, batch_example_s) in enumerate(
                self.test_loader):
            batch_true = batch_example_s.tolist()
            test_true_int.extend(batch_true)

            test_y_pred = model(batch_test_x0.to(self.device), batch_test_x1.to(self.device),
                                batch_test_llm.to(self.device), batch_test_llm_mask.to(self.device))[0]
            test_y_pred = test_y_pred.cpu().detach().numpy().reshape(-1, trait_num)
            if test_y_pred.shape[0] > 1:
                test_pred_i = test_y_pred.squeeze()
            else:
                test_pred_i = np.reshape(test_y_pred, (1, -1))
            test_pred_int.extend(test_pred_i)
        test_true_dict = separate_and_rescale_attributes_for_scoring_feedback(test_true_int)
        test_pred_dict = separate_and_rescale_attributes_for_scoring_feedback(test_pred_int)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], dev_true_dict[key]) for key in
                          dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], test_true_dict[key]) for key in
                           test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            file_path = os.path.join(dir, f"{model_name}-" + str(self.prompt_id) + ".ckpt")
            torch.save(model, file_path)  # save the best model
            print("Save best model to ", file_path)
        if self.test_kappa_mean > self.test_best_value:
            self.test_best_value = self.test_kappa_mean
            self.best_test_epoch = epoch
        if print_info:
            self.print_info(epoch, res_log)

    def evaluate_base(self, model, epoch, print_info=False, res_log=None, model_name=''):
        dev_pred_int = []
        dev_true_int = []
        for step, (batch_dev_x0, batch_dev_x1, batch_dev_f, batch_example_s) in enumerate(self.dev_loader):
            batch_true = batch_example_s.tolist()
            dev_true_int.extend(batch_true)

            dev_y_pred = model(batch_dev_x0.to(self.device), batch_dev_x1.to(self.device),
                               batch_dev_f.to(self.device))
            dev_y_pred = dev_y_pred.cpu()
            if dev_y_pred.shape[0] > 1:
                dev_pred_i = dev_y_pred.detach().numpy().squeeze()
            else:
                dev_pred_i = dev_y_pred.detach().numpy()
            dev_pred_int.extend(dev_pred_i)

        dev_true_dict = separate_and_rescale_attributes_for_scoring(dev_true_int, self.prompt_id)
        dev_pred_dict = separate_and_rescale_attributes_for_scoring(dev_pred_int, self.prompt_id)

        test_pred_int = []
        test_true_int = []
        for step, (batch_test_x0, batch_test_x1, batch_test_f, batch_example_s) in enumerate(self.test_loader):
            batch_true = batch_example_s.tolist()
            test_true_int.extend(batch_true)

            test_y_pred = model(batch_test_x0.to(self.device), batch_test_x1.to(self.device),
                                batch_test_f.to(self.device))
            test_y_pred = test_y_pred.cpu()
            if test_y_pred.shape[0] > 1:
                test_pred_i = test_y_pred.detach().numpy().squeeze()
            else:
                test_pred_i = test_y_pred.detach().numpy().squeeze(axis=2)
            test_pred_int.extend(test_pred_i)
        test_true_dict = separate_and_rescale_attributes_for_scoring(test_true_int, self.prompt_id)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred_int, self.prompt_id)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], dev_true_dict[key]) for key in
                          dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], test_true_dict[key]) for key in
                           test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            file_path = os.path.join(dir, f"{model_name}-checkpoint_best-" + str(self.prompt_id) + ".ckpt")
            torch.save(model, file_path)  # save the best model
            print("Save best model to ", file_path)
        if self.test_kappa_mean > self.test_best_value:
            self.test_best_value = self.test_kappa_mean
            self.best_test_epoch = epoch
        if print_info:
            self.print_info(epoch, res_log)

    def print_info(self, epoch, res_log):
        print('CURRENT EPOCH: {}'.format(epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        res_log.write('CURRENT EPOCH: {} \n'.format(epoch))
        res_log.write('[DEV] AVG QWK: {} \n'.format(round(self.dev_kappa_mean, 3)))
        res_log.flush()
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(att, round(self.kappa_dev[att], 3)))
            res_log.write('[DEV] {} QWK: {} \n'.format(att, round(self.kappa_dev[att], 3)))
            res_log.flush()
        res_log.write('------------------------ \n')
        res_log.flush()
        print(
            '------------------------')
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        res_log.write('[TEST] AVG QWK: {} \n'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(att, round(self.kappa_test[att], 3)))
            res_log.write('[TEST] {} QWK: {} \n'.format(att, round(self.kappa_test[att], 3)))
            res_log.flush()

        print(
            '------------------------')
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        res_log.write(
            '[BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))

        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
            res_log.write('[BEST TEST] {} QWK: {} \n'.format(att, round(self.best_test_kappa_set[att], 3)))
            res_log.flush()
        res_log.write('------------------------ \n')
        print('[THE BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.test_best_value, 3), self.best_test_epoch))
        res_log.write(
            '[THE BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.test_best_value, 3), self.best_test_epoch))
        res_log.flush()
        print(
            '--------------------------------------------------------------------------------------------------'
            '------------------------')

    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print(
            '---------------------------------------------------------------------------------------------------'
            '-----------------------')

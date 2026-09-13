# -*- coding: utf-8 -*-
import argparse
import torch
import data_prepare
import time
import warnings
import random
import torch.utils.data as Data
from networks.Trait_Scorer import MultiLayer_Scorer
from evaluator.evaluator_core import Evaluator_normal_feature
from utils import *

from transformers import RobertaTokenizer
from torch.optim import AdamW
from losses import hidden_sim_loss, SC_loss, total_loss, orthogonality_loss

warnings.filterwarnings('ignore')
logger = get_logger("Train...")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_percentage = (trainable_params / total_params) * 100

    print(
        f"Total parameters: {total_params} || Trainable parameters: {trainable_params} || Trainable parameters percentage: {trainable_params_percentage:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="GenMAES model")
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of texts in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for layers')
    parser.add_argument('--prompt_id', type=int, default=5, help='prompt id of essay set')
    parser.add_argument('--fold_id', type=int, default=0, help='fold of essay set')
    parser.add_argument('--gpu', type=int, default=0, help='GPU Id')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model_type', type=str, default='robert', help='encoder type')

    args = parser.parse_args()
    learning_rate = args.learning_rate
    fold = args.fold_id
    gpu_id = args.gpu

    set_seed(args.seed)
    model_path = './models/robert'
    tokenizer = RobertaTokenizer.from_pretrained(model_path, sep_token='[SEP]')

    train = './asap-dataset/fold_' + str(fold) + '/train.csv'
    dev = './asap-dataset/fold_' + str(fold) + '/dev.csv'
    test = './asap-dataset/fold_' + str(fold) + '/test.csv'
    datapaths = [train, dev, test]
    prompt_id = args.prompt_id

    res_log = open(f'./res_logs/DPRES-{prompt_id}-F{fold}.txt', 'w', encoding='utf-8')

    (X_train_ids, X_train_mask, Y_train), \
    (X_dev_ids, X_dev_mask, Y_dev), \
    (X_test_ids, X_test_mask, Y_test) = \
        data_prepare.prepare_sentence_data_for_ASAP(datapaths, prompt_id=prompt_id, tokenizer=tokenizer)

    features_dev = [X_dev_ids, X_dev_mask, Y_dev]
    features_test = [X_test_ids, X_test_mask, Y_test]

    train_dataset = Data.TensorDataset(torch.LongTensor(X_train_ids), torch.LongTensor(X_train_mask),
                                       torch.Tensor(Y_train))

    loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    logger.info("----------------------------------------------------")

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    logger.info("Model building")
    model = MultiLayer_Scorer(args, model_path=model_path, trait_num=Y_train.shape[-1])
    print_trainable_params(model)
    model.to(device)

    evl = Evaluator_normal_feature(args.prompt_id, features_dev, Y_dev, features_test, Y_test, device)
    with torch.no_grad():
        evl.evaluate(model, 0, True, res_log=res_log, model_name=f'Fold{fold}')

    logger.info("Train model")

    optimizer = AdamW(model.parameters(), lr=learning_rate)  # AdamW

    for ii in range(args.num_epochs):
        print('Epoch %s/%s' % (str(ii + 1), args.num_epochs))
        res_log.write(('Epoch %s/%s \n' % (str(ii + 1), args.num_epochs)))
        start_time = time.time()
        model.train()
        for step, (ids, masks, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            Y_predict, encoder_features, shared_repr, trait_residuals, essay_features = model(ids.to(device),
                                                                                              masks.to(device))

            orth_loss = orthogonality_loss(shared_repr, trait_residuals)
            hsl = hidden_sim_loss(essay_features)
            scl = SC_loss(encoder_features, batch_y)
            lam = 1 - ((ii + 1) / args.num_epochs) ** 2
            cl_loss = (1 - lam) * scl + lam * hsl

            task_loss = total_loss(Y_predict, batch_y.to(device))
            loss = task_loss + 0.01 * cl_loss + 0.1 * orth_loss

            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print('epoch: {} ||'.format(ii + 1), 'step: {} ||'.format(step), 'loss: %.5f' % (loss.item()))

        tt_time = time.time() - start_time
        print("Training one epoch in %.3f s" % tt_time)

        model.eval()
        with torch.no_grad():
            evl.evaluate(model, ii + 1, True, res_log=res_log, model_name=f'Fold{fold}')

    evl.print_final_info()


if __name__ == '__main__':
    main()

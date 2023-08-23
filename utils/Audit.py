#coding:utf-8
import argparse
import sys
import torch
from utils.MIA import mia
import os
import pickle
from scipy.stats import ks_2samp, ttest_ind
import numpy as np
from utils.Log import log_creater
from utils.Risk import riskScore, riskScore2
import time
import torch.nn.functional as F

metrics = ['correctness', 'confidence', 'entropy']

def get_member_ratio(datadf, thre_cnt=1, skip=[], mode='ks'):
    sample2mem = []
    for idx, row in datadf.iterrows():
        sample2mem.append(np.sum([row[m] for m in (set(metrics) - set(skip))]))

    members_bool = np.asarray(sample2mem) >= thre_cnt

    if mode == 'ks':
        _, pvalue = ks_2samp(members_bool, [1 for i in range(10000)], mode='asymp')
    elif mode == 't':
        if members_bool.mean() == 1:
            return 1, 1
        else:
            _, pvalue = ttest_ind(members_bool, [1 for i in range(len(members_bool))], equal_var=True,
                                  nan_policy='raise')

    return np.sum(members_bool) / len(datadf), pvalue

def api(args, model, queryDataset, member_gt, calDataset, logger):
    queryTestDataLoader = queryDataset.test_dataloader()
    calTrainDataLoader = calDataset.train_dataloader()
    calTestnDataLoader = calDataset.test_dataloader()

    model.eval()
    # modelCal.eval()

    calTrainPred = []
    calTrainY = []
    calTestPred = []
    calTestY = []
    queryPred = []
    queryY = []

    # first, get query_output
    with torch.no_grad():
        for X, Y in queryTestDataLoader:
            X = X.to(args.device)
            Y = Y.to(args.device)
            if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

            if 'MNIST' in args.root:
                out_Y = torch.exp(model(X))
            elif 'COVIDx' in args.root:
                out_Y = torch.exp(F.log_softmax(model(X), dim=-1))
            else:
                out_Y = torch.exp(F.log_softmax(model(X), dim=-1))
            queryPred.append(out_Y)
            queryY.append(Y)
    queryY = torch.cat(queryY).detach().cpu().numpy()
    queryPred = torch.cat(queryPred).detach().cpu().numpy()

    # then， get calibration set output
    with torch.no_grad():
        for X, Y in calTrainDataLoader:
            X = X.to(args.device)
            Y = Y.to(args.device)
            if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

            if 'MNIST' in args.root:
                out_Y = torch.exp(model(X))
            elif 'COVIDx' in args.root:
                out_Y = torch.exp(F.log_softmax(model(X), dim=-1))
            else:
                out_Y = torch.exp(F.log_softmax(model(X), dim=-1))
            calTrainPred.append(out_Y)
            calTrainY.append(Y)
    calTrainY = torch.cat(calTrainY).detach().cpu().numpy()
    calTrainPred = torch.cat(calTrainPred).detach().cpu().numpy()

    with torch.no_grad():
        for X, Y in calTestnDataLoader:
            X = X.to(args.device)
            Y = Y.to(args.device)
            if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

            if 'MNIST' in args.root:
                out_Y = torch.exp(model(X))
            elif 'COVIDx' in args.root:
                out_Y = torch.exp(F.log_softmax(model(X), dim=-1))
            else:
                out_Y = torch.exp(F.log_softmax(model(X), dim=-1))
            calTestPred.append(out_Y)
            calTestY.append(Y)
    calTestY = torch.cat(calTestY).detach().cpu().numpy()
    calTestPred = torch.cat(calTestPred).detach().cpu().numpy()

    # run MIA
    MIA = mia(calTrainPred,
              calTrainY,
              calTestPred,
              calTestY,
              queryPred,
              queryY,
              num_classes=args.nclass)

    MIA_results = MIA._run_mia()

    t, pv = get_member_ratio(MIA_results['query_values_bin'], skip=['modified entropy'], mode='t')
    EMA_res = np.around(pv, decimals=2)
    risk_score = riskScore2(MIA_results['query_values_bin'], member_gt)
    return t, pv, EMA_res, risk_score

def main(args):
    sys.path.append(args.root)
    from Dataset import DataModule, CONFIG
    from Model import get_model, get_student_model

    args.mode = 'query'
    queryDataset = DataModule(dir=args.root, args=args, batch_size = args.KP_infer_batch_size, num_workers=args.num_workers)
    queryTestDataLoader = queryDataset.test_dataloader()
    args.mode = 'cal'
    calDataset = DataModule(dir=args.root, args=args, batch_size = args.KP_infer_batch_size, num_workers=args.num_workers)
    calTrainDataLoader = calDataset.train_dataloader()
    calTestnDataLoader = calDataset.test_dataloader()

    if ('student' in args.model2audit) | ('best_model.pth_KD' in args.model2audit) | ('best_model.pth_KP' in args.model2audit):
        modelBase = get_student_model().to(args.device)
    else:
        modelBase = get_model().to(args.device)
    if 'student' in args.model2cal:
        modelCal = get_student_model().to(args.device)
    else:
        modelCal = get_model().to(args.device)

    # load model weight
    modelBase.load_state_dict(torch.load(f'{args.root}/{args.model2audit}', map_location=torch.device(args.device)))
    #modelCal.load_state_dict(torch.load(f'{args.root}/{args.model2cal}', map_location=torch.device(args.device)))

    modelBase.eval()
    #modelCal.eval()

    calTrainPred = []
    calTrainY = []
    calTestPred = []
    calTestY = []
    queryPred = []
    queryY = []

    # first, get query_output
    start = time.time()
    with torch.no_grad():
        for X, Y in queryTestDataLoader:
            X = X.to(args.device)
            Y = Y.to(args.device)
            if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

            if 'MNIST' in args.root:
                out_Y = torch.exp(modelBase(X))
            elif 'COVIDx' in args.root:
                out_Y = torch.exp(F.log_softmax(modelBase(X), dim=-1))
            else:
                out_Y = torch.exp(F.log_softmax(modelBase(X), dim=-1))
            queryPred.append(out_Y)
            queryY.append(Y)
    queryY = torch.cat(queryY).detach().cpu().numpy()
    queryPred = torch.cat(queryPred).detach().cpu().numpy()
    logger.info('>> finished inferring query dataset')
    end = time.time()
    #print(f'>> time cost: {end-start}')

    # then， get calibration set output
    start = time.time()
    with torch.no_grad():
        for X, Y in calTrainDataLoader:
            X = X.to(args.device)
            Y = Y.to(args.device)
            if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

            if 'MNIST' in args.root:
                out_Y = torch.exp(modelBase(X))
            elif 'COVIDx' in args.root:
                out_Y = torch.exp(F.log_softmax(modelBase(X), dim=-1))
            else:
                out_Y = torch.exp(F.log_softmax(modelBase(X), dim=-1))
            calTrainPred.append(out_Y)
            calTrainY.append(Y)
    calTrainY = torch.cat(calTrainY).detach().cpu().numpy()
    calTrainPred = torch.cat(calTrainPred).detach().cpu().numpy()
    end = time.time()
    #print(f'>> time cost: {end - start}')

    start = time.time()
    with torch.no_grad():
        for X, Y in calTestnDataLoader:
            X = X.to(args.device)
            Y = Y.to(args.device)
            if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

            if 'MNIST' in args.root:
                out_Y = torch.exp(modelBase(X))
            elif 'COVIDx' in args.root:
                out_Y = torch.exp(F.log_softmax(modelBase(X), dim=-1))
            else:
                out_Y = torch.exp(F.log_softmax(modelBase(X), dim=-1))
            calTestPred.append(out_Y)
            calTestY.append(Y)
    calTestY = torch.cat(calTestY).detach().cpu().numpy()
    calTestPred = torch.cat(calTestPred).detach().cpu().numpy()
    logger.info('>> finished inferring cal train dataset and cal test dataset')

    logger.info(f'>> query output: {queryPred.shape}, cal train output: {calTrainPred.shape}, cal test output: {calTestPred.shape}')
    end = time.time()
    #print(f'>> time cost: {end - start}')

    # run MIA
    start = time.time()
    MIA = mia(calTrainPred,
              calTrainY,
              calTestPred,
              calTestY,
              queryPred,
              queryY,
              num_classes=args.nclass)

    MIA_results = MIA._run_mia()
    logger.info('>> MIA finished')
    end = time.time()
    #print(f'>> time cost: {end - start}')

    if not os.path.exists(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/thres'):
        os.makedirs(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/cal_set')
        os.makedirs(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/query_set')
        os.makedirs(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/thres')

    MIA_results['cal_values_bin'].to_csv(
        f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/cal_set/binarized.csv', index=False)
    MIA_results['cal_values'].to_csv(
        f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/cal_set/continuous.csv', index=False)

    MIA_results['query_values_bin'].to_csv(
        f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/query_set/binarized.csv', index=False)
    MIA_results['query_values'].to_csv(
        f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/query_set/continuous.csv', index=False)

    pickle.dump(MIA_results['thresholds'], open(
        f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/thres/thres.pkl', 'wb'))
    logger.info('>> results saved')

    t, pv = get_member_ratio(MIA_results['query_values_bin'], skip=['modified entropy'], mode='t')
    EMA_res = np.around(pv, decimals=2)
    member_gt = CONFIG[args.query_label]['QUERY_MEMBER']
    risk_score = riskScore2(MIA_results['query_values_bin'], member_gt)
    logger.info(f'>> test value: {t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')

def parser():
    """

    :return: args
    """

    parser = argparse.ArgumentParser(prog='Audit')
    parser.add_argument('--root',
                          default='../template/MNIST',
                          help='root dir to the project')
    parser.add_argument('--query_label',
                        default='EXP1',
                        help='label of the query data, defined in Dataset.py/Config')
    parser.add_argument('--cal_label',
                        default='CAL1',
                        help='label of the calibration data, defined in Dataset.py/Config')
    parser.add_argument('--cal_test_label',
                        default='CALTEST1',
                        help='label of the calibration test data, defined in Dataset.py/Config')
    parser.add_argument('--test_label',
                        default='TEST1',
                        help='label of the test data, defined in Dataset.py/Config')
    parser.add_argument('--model2audit',
                        default='./models/base/best_model.pth',
                        help='relative path of model to be auditted to the root')
    parser.add_argument('--model2cal',
                       default='./models/cal/best_model.pth',
                       help='relative path of the calibration model to the root')
    parser.add_argument('--device',
                        default='cuda:0')
    parser.add_argument('--KP_infer_batch_size',
                        type=int,
                        default=1024,
                        help='batch size for inference during membership attack')
    parser.add_argument('--nclass',
                        type=int,
                        default=10,
                        help='number of classes')
    parser.add_argument('--num_workers',
                        type=int,
                        default=5,
                        help='number of num_workers')

    args = parser.parse_args()
    return args

def one_command_api(args):
    global logger
    args.postfix = f'cal:{args.cal_label}_calTest:{args.cal_test_label}_test:{args.test_label}_query:{args.query_label}'
    if not os.path.exists(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}'):
        os.makedirs(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}')
    logger = log_creater(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/log.log')
    logger.info(args)
    main(args)

if __name__ == '__main__':
    args = parser()
    args.postfix = f'cal:{args.cal_label}_calTest:{args.cal_test_label}_test:{args.test_label}_query:{args.query_label}'
    if not os.path.exists(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}'):
        os.makedirs(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}')
    logger = log_creater(f'{args.root}/{args.model2audit}_MIA_@{args.postfix}/log.log')
    logger.info(args)
    main(args)
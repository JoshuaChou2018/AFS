import argparse
import sys
import torch
from utils.KDloss import SoftTarget
from utils.Metric import AverageMeter, accuracy, Performance
import os
import numpy as np
from tqdm import tqdm
from utils.Log import log_creater
import utils.Audit as Audit
import torch.nn as nn
import time

def train(snet, tnet, criterionCls, criterionKD, trainDataLoader, optimizer, args, queryDataset, member_gt, calDataset):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    risk_losses = AverageMeter()
    total_losses = AverageMeter()
    snet.train()
    tnet.eval()
    with tqdm(total=len(trainDataLoader)) as t:
        for X, Y in tqdm(trainDataLoader):
            X = X.to(args.device)
            Y = Y.to(args.device)

            if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

            #print(X)
            snet_pred = snet(X)
            tnet_pred = tnet(X)

            #print(Y, tnet_pred, snet_pred)

            cls_loss = criterionCls(snet_pred, Y)
            kd_loss = criterionKD(snet_pred, tnet_pred.detach())
            cls_losses.update(cls_loss.item(), X.size(0))
            kd_losses.update(kd_loss.item(), X.size(0))

            loss = cls_loss + kd_loss* args.lambda_kd

            #if loss < 1:
            #    args.add_risk_loss = True
            #    print(f'>> start using risk loss')
            #else:
            #    args.add_risk_loss = False


            if args.add_risk_loss == 1:
                risk_loss = torch.tensor(0.0).to(args.device)
                queryTestDataLoader = queryDataset.test_dataloader()
                for _X, _Y in queryTestDataLoader:
                    _X = _X.to(args.device)
                    _Y = _Y.to(args.device)
                    if 'PathMNIST' in args.root:
                        _Y = _Y.squeeze().long()
                    out_Y = snet(_X)
                    partial_risk_loss = torch.nn.CrossEntropyLoss().to(args.device)(out_Y, _Y)
                    risk_loss += partial_risk_loss

                ## using Audit.api requires more time for training
                #t, pv, EMA_res, risk_score = Audit.api(args, model, queryDataset, member_gt, calDataset)
                #risk_loss = risk_score

                risk_loss = torch.tensor(1.0).to(args.device) / risk_loss # same performance, strongly correlated

                #risk_loss = risk_loss
                risk_losses.update(risk_loss.item(), _X.size(0))
                loss = loss + risk_loss*torch.tensor(args.lambda_risk).to(args.device)

            total_losses.update(loss.item(), X.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(
                cls_losses='{:05.8f}'.format(cls_losses.avg),
                kd_losses='{:05.8f}'.format(kd_losses.avg),
                risk_losses='{:05.8f}'.format(risk_losses.avg),
                total_losses='{:05.8f}'.format(total_losses.avg),
            )
            t.update()

def test(snet, tnet, criterionCls, criterionKD, testDataLoader, args):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    correct = 0

    snet.eval()
    tnet.eval()

    total_pred = []
    total_y = []

    with torch.no_grad():
        with tqdm(total=len(testDataLoader)) as t:
            for X, Y in tqdm(testDataLoader):
                X = X.to(args.device)
                Y = Y.to(args.device)

                if 'PathMNIST' in args.root:
                    Y = Y.squeeze().long()

                snet_pred = snet(X)
                tnet_pred = tnet(X)

                cls_loss = criterionCls(snet_pred, Y)
                kd_loss = criterionKD(snet_pred, tnet_pred.detach()) * args.lambda_kd

                pred = snet_pred.data.max(1)[1]
                correct += pred.eq(Y.view(-1)).sum().item()

                cls_losses.update(cls_loss.item(), X.size(0))
                kd_losses.update(kd_loss.item(), X.size(0))

                t.set_postfix(
                    cls_losses='{:05.8f}'.format(cls_losses.avg),
                    kd_losses='{:05.8f}'.format(kd_losses.avg),
                )
                t.update()

                total_pred += pred
                total_y += Y.view(-1)

    test_acc = correct / len(testDataLoader.dataset)
    stat = Performance(total_pred, total_y)

    return test_acc, stat

def save_best_ckpt(model, args):
    if not os.path.exists(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}'):
        os.makedirs(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}')
    torch.save(model.state_dict(
        ), f'{args.root}/{args.teacher_model}_KP_@{args.postfix}/student.pth')

def save_last_ckpt(model, args):
    if not os.path.exists(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}'):
        os.makedirs(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}')
    torch.save(model.state_dict(
        ), f'{args.root}/{args.teacher_model}_KP_@{args.postfix}/last_model.pth')

def main(args):
    sys.path.append(args.root)
    from Dataset import DataModule, CONFIG
    from Model import get_teacher_model, get_student_model

    # init model
    tnet = get_teacher_model().to(args.device)
    snet = get_student_model().to(args.device)

    tnet.load_state_dict(torch.load(f'{args.root}/{args.teacher_model}', map_location=torch.device(args.device)))
    tnet.eval()

    if 'MNIST' in args.root:
        criterionCls = torch.nn.NLLLoss().to(args.device)
    elif 'PathMNIST' in args.root:
        criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    elif 'COVIDx' in args.root:
        criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    else:
        criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    criterionKD = SoftTarget(args.T)
    optimizer = torch.optim.Adam(snet.parameters(),
                                lr=args.lr)

    args.mode = 'base'
    args.base_label = args.KD_label
    baseDataset = DataModule(dir=args.root, args = args, batch_size = args.train_batch_size, num_workers=args.num_workers)
    baseTrainDataLoader = baseDataset.train_dataloader()
    baseTestDataLoader = baseDataset.test_dataloader()
    args.mode = 'query'
    queryDataset = DataModule(dir=args.root, args=args, batch_size = args.KP_infer_batch_size, num_workers=args.num_workers)
    member_gt = CONFIG[args.query_label]['QUERY_MEMBER']
    args.mode = 'cal'
    calDataset = DataModule(dir=args.root, args=args, batch_size = args.KP_infer_batch_size, num_workers=args.num_workers)

    logger.info('>> start KP')
    save_metric_best = 0
    for epoch in range(args.epochs):
        logger.info(f'>> epoch {epoch}')
        logger.info(f'>> train models')
        train(snet, tnet, criterionCls, criterionKD, baseTrainDataLoader, optimizer, args, queryDataset, member_gt, calDataset)
        logger.info(f'>> test models')
        test_acc, stat = test(snet, tnet, criterionCls, criterionKD, baseTestDataLoader, args)
        logger.info(f'>> snet test acc: {test_acc}')
        logger.info(f">> snet test stat: {','.join([str(_) for _ in stat])}")

        logger.info(f'>> evaluate membership attack on snet after training')
        _t, pv, EMA_res, risk_score = Audit.api(args, snet, queryDataset, member_gt, calDataset, logger)
        logger.info(f'>> test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')

        # save model
        if save_metric_best < test_acc:
            save_metric_best = test_acc
            logger.info(f'>> saving best snet model')
            save_best_ckpt(snet, args)

        save_last_ckpt(snet, args)

def parser():
    """

    :return: args
    """

    parser = argparse.ArgumentParser(prog='Knowledge Purification (KP)')
    parser.add_argument('--root',
                          default='../template/MNIST',
                          help='root dir to the project')
    parser.add_argument('--expname',
                        default='EXP1',
                        help='name of exp, will affect the path and dataset splitting')
    parser.add_argument('--teacher_model',
                        default='./models/EXP1/base/best_model.pth',
                        help='relative path of model to be distilled to the root')
    parser.add_argument('--KD_label',
                        default='KD0.25',
                        help='the name of base dataset used for KD, should be defined in CONFIG')
    parser.add_argument('--test_label',
                        default='TEST1',
                        help='label of the test data, defined in Dataset.py/Config')
    parser.add_argument('--cal_label',
                        default='CAL1',
                        help='label of the calibration data, defined in Dataset.py/Config')
    parser.add_argument('--cal_test_label',
                        default='CALTEST1',
                        help='label of the calibration test data, defined in Dataset.py/Config')
    parser.add_argument('--query_label',
                        default='QO1',
                        help='label of the query data, defined in Dataset.py/Config, here the query dataset should overlap with training dataset')
    parser.add_argument('--add_risk_loss',
                        type=int,
                        default=1,
                        help='1: will add risk loss when running KP, 0: same as pure KD')
    parser.add_argument('--nclass',
                        type=int,
                        default=10,
                        help='number of classes')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--KP_infer_batch_size',
                        type=int,
                        default=128,
                        help='batch size for inference during membership attack')
    parser.add_argument('--device',
                        default='cuda:0')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='number of epochs')
    parser.add_argument('--T',
                        type=float,
                        default=4.0,
                        help='temperature for ST')
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='initial learning rate')
    parser.add_argument('--lambda_kd',
                        type=float,
                        default=1,
                        help='trade-off parameter for kd loss')
    parser.add_argument('--lambda_risk',
                        type=float,
                        default=10,
                        help='trade-off parameter for risk loss')
    parser.add_argument('--num_workers',
                        type=int,
                        default=5,
                        help='number of num_workers')



    args = parser.parse_args()
    return args

def one_command_api(args):
    global logger
    args.postfix = f'KD:{args.KD_label}_query:{args.query_label}'
    if not os.path.exists(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}'):
        os.makedirs(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}')
    logger = log_creater(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}/log.log')
    logger.info(args)
    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'time: {end - start}')

if __name__ == '__main__':
    args = parser()
    args.postfix = f'KD:{args.KD_label}_query:{args.query_label}'
    if not os.path.exists(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}'):
        os.makedirs(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}')
    logger = log_creater(f'{args.root}/{args.teacher_model}_KP_@{args.postfix}/log.log')
    logger.info(args)
    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'time: {end-start}')

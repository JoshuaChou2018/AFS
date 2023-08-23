import argparse
import sys
import torch
from utils.KDloss import SoftTarget
from utils.Metric import AverageMeter, accuracy, Performance
import os
import numpy as np
from tqdm import tqdm
from utils.Log import log_creater
import time
import torch.nn as nn

def train(snet, tnet, criterionCls, criterionKD, trainDataLoader, optimizer, args):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    total_losses = AverageMeter()
    snet.train()
    tnet.eval()
    with tqdm(total=len(trainDataLoader)) as t:
        for X, Y in tqdm(trainDataLoader):
            X = X.to(args.device)
            Y = Y.to(args.device)

            if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

            snet_pred = snet(X)
            tnet_pred = tnet(X)

            #print(Y, tnet_pred, snet_pred)
            #print(snet_pred, Y)
            cls_loss = criterionCls(snet_pred, Y)
            kd_loss = criterionKD(snet_pred, tnet_pred.detach()) * args.lambda_kd
            loss = cls_loss + kd_loss

            cls_losses.update(cls_loss.item(), X.size(0))
            kd_losses.update(kd_loss.item(), X.size(0))
            total_losses.update(loss.item(), X.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(
                cls_losses='{:05.8f}'.format(cls_losses.avg),
                kd_losses='{:05.8f}'.format(kd_losses.avg),
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
                loss = cls_loss + kd_loss

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
    if not os.path.exists(f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}'):
        os.makedirs(f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}')
    torch.save(model.state_dict(
        ), f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}/student.pth')

def save_last_ckpt(model, args):
    if not os.path.exists(f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}'):
        os.makedirs(f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}')
    torch.save(model.state_dict(
        ), f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}/last_model.pth')

def main(args):
    sys.path.append(args.root)
    from Dataset import DataModule
    from Model import get_teacher_model, get_student_model

    # init model
    tnet = get_teacher_model().to(args.device)
    snet = get_student_model().to(args.device)

    tnet.load_state_dict(torch.load(f'{args.root}/{args.teacher_model}', map_location=torch.device(args.device)))
    tnet.eval()

    if 'MNIST' in args.root and 'PathMNIST' not in args.root:
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
    baseDataset = DataModule(dir=args.root, args = args, batch_size = args.batch_size)
    baseTrainDataLoader = baseDataset.train_dataloader()
    baseTestDataLoader = baseDataset.test_dataloader()

    logger.info('>> start KD')
    save_metric_best = 0
    for epoch in range(args.epochs):
        logger.info(f'>> epoch {epoch}')
        logger.info(f'>> train models')
        train(snet, tnet, criterionCls, criterionKD, baseTrainDataLoader, optimizer, args)
        logger.info(f'>> test models')
        test_acc, stat = test(snet, tnet, criterionCls, criterionKD, baseTestDataLoader, args)
        logger.info(f'>> snet test acc: {test_acc}')
        logger.info(f">> snet test stat: {','.join([str(_) for _ in stat])}")

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

    parser = argparse.ArgumentParser(prog='Knowledge Distillation (KD)')
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
    parser.add_argument('--device',
                        default='cuda:0')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
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



    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser()
    if not os.path.exists(f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}'):
        os.makedirs(f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}')
    logger = log_creater(f'{args.root}/{args.teacher_model}_KD_@{args.KD_label}/log.log')
    logger.info(args)
    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'time: {end-start}')
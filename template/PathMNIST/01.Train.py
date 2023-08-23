from Dataset import DataModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import csv
import argparse
from Model import get_model, get_student_model
from tqdm import tqdm
import sys
from Model import Net


sys.path.append('../..')
from utils.Log import log_creater

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

class TrainModule:
    def __init__(self, dataset, criterion, max_epoch, mode='base', ckpt_name='', args=None):

        self.dataset = dataset
        self.train_loader = dataset.train_dataloader()
        self.test_loader = dataset.test_dataloader()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if args.pure_student == True:
            self.model = get_student_model().to(self.device)
        else:
            self.model = get_model().to(self.device)
        self.criterion = criterion
        self.max_epoch = max_epoch

        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        self.res = []

        self.mode = mode
        self.ckpt_name = ckpt_name
        self.base_label = args.base_label

        self.alpha = 1

        self.best_acc = 0
        self.args = args

    def Performance(self, predict, y_test):
        from math import sqrt
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        tp_idx = []
        tn_idx = []
        fp_idx = []
        fn_idx = []

        for i in range(len(y_test)):
            if (y_test[i] == 1):
                if (predict[i] == 1):
                    tp += 1
                    tp_idx.append(i)
                else:
                    fn += 1
                    fn_idx.append(i)
            if (y_test[i] == 0):
                if (predict[i] == 1):
                    fp += 1
                    fp_idx.append(i)
                else:
                    tn += 1
                    tn_idx.append(i)

        tpr, tnr, fpr, acc, mcc, fdr, f1 = -1, -1, -1, -1, -1, -1, -1
        try:
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            fpr = fp / (fp + tn)
            acc = (tp + tn) / (tp + tn + fp + fn)
            mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            fdr = fp / (fp + tp)
            f1 = 2 * tp / (2 * tp + fp + fn)
        except Exception:
            pass
        return [tp, tn, fp, fn, tpr, tnr, fpr, acc, mcc, fdr, f1]

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        loss_avg = RunningAverage()

        total_pred = []
        total_y = []

        with torch.no_grad():
            with tqdm(total=len(self.test_loader)) as t:
                for X, Y in tqdm(self.test_loader):
                    X = X.to(self.device)
                    Y = Y.to(self.device)

                    output = self.model(X)
                    Y = Y.squeeze().long()

                    class_loss = self.criterion(output, Y).item()
                    pred = output.data.max(1)[1]
                    correct += pred.eq(Y.view(-1)).sum().item()
                    test_loss += class_loss

                    loss_avg.update(class_loss)
                    t.set_postfix(
                        loss_avg='{:05.8f}'.format(loss_avg()),
                        total_loss='{:05.8f}'.format(class_loss),
                    )
                    t.update()

                    total_pred += pred
                    total_y += Y.view(-1)

        test_loss /= len(self.test_loader)
        correct /= len(self.test_loader.dataset)
        stat = self.Performance(total_pred, total_y)

        return test_loss, correct, stat

    def train(self):
        self.model.train()
        correct = 0
        train_loss = 0
        loss_avg = RunningAverage()
        with tqdm(total=len(self.train_loader)) as t:
            for X, Y in tqdm(self.train_loader):
                X = X.to(self.device)
                Y = Y.to(self.device)

                # Training pass
                self.optimizer.zero_grad()

                output = self.model(X)
                Y = Y.squeeze().long()
                loss = self.criterion(output, Y)

                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                pred = output.data.max(1)[1]
                correct += pred.eq(Y.view(-1)).sum().item()

                loss_avg.update(loss.item())
                t.set_postfix(
                    loss_avg='{:05.8f}'.format(loss_avg()),
                    total_loss='{:05.8f}'.format(loss.item()),
                )
                t.update()

        train_loss /= len(self.train_loader)
        correct /= len(self.train_loader.dataset)
        test_loss, test_acc, stat = self.test()

        self.args.logger.info('Train set: Average loss: {:.4f}, Average acc: {:.4f}'.format(train_loss, correct))
        self.args.logger.info('Test set: Average loss: {:.4f}, Average acc: {:.4f}, stat: {}'.format(test_loss, test_acc, ','.join([str(_) for _ in stat])))
        self.res.append([train_loss, correct, test_loss, test_acc])

        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.save_best_ckpt()
            self.args.logger.info('>> saving best model')

    def save_ckpt(self, epoch):
        if epoch % 10 == 0 or epoch == self.max_epoch:
            if args.pure_student == False:
                if not os.path.exists(f'models/{args.expname}/{self.base_label}'):
                    os.makedirs(f'models/{args.expname}/{self.base_label}')
                torch.save(self.model.state_dict(
                ), f'models/{args.expname}/{self.base_label}/{self.ckpt_name}training_epoch{epoch}.pth')
            else:
                if not os.path.exists(f'models/{args.expname}/{self.base_label}_pure_student'):
                    os.makedirs(f'models/{args.expname}/{self.base_label}_pure_student')
                torch.save(self.model.state_dict(
                ), f'models/{args.expname}/{self.base_label}_pure_student/{self.ckpt_name}training_epoch{epoch}.pth')

    def save_last_ckpt(self):
        if args.pure_student == False:
            if not os.path.exists(f'models/{args.expname}/{self.base_label}'):
                os.makedirs(f'models/{args.expname}/{self.base_label}')
            torch.save(self.model.state_dict(
                ), f'models/{args.expname}/{self.base_label}/last_model.pth')
        else:
            if not os.path.exists(f'models/{args.expname}/{self.base_label}_pure_student'):
                os.makedirs(f'models/{args.expname}/{self.base_label}_pure_student')
            torch.save(self.model.state_dict(
                ), f'models/{args.expname}/{self.base_label}_pure_student/last_model.pth')

    def save_best_ckpt(self):
        if args.pure_student == False:
            if not os.path.exists(f'models/{args.expname}/{self.base_label}'):
                os.makedirs(f'models/{args.expname}/{self.base_label}')
            torch.save(self.model.state_dict(
                ), f'models/{args.expname}/{self.base_label}/best_model.pth')
        else:
            if not os.path.exists(f'models/{args.expname}/{self.base_label}_pure_student'):
                os.makedirs(f'models/{args.expname}/{self.base_label}_pure_student')
            torch.save(self.model.state_dict(
                ), f'models/{args.expname}/{self.base_label}_pure_student/best_model.pth')

    def save_log(self):
        if args.pure_student == False:
            logname = f'models/{args.expname}/{self.base_label}/metrics.log'
        else:
            logname = f'models/{args.expname}/{self.base_label}_pure_student/metrics.log'
        with open(logname, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerow(
                ["Train loss", "Train acc", "Test loss", "Test acc"])
            for row in self.res:
                writer.writerow(row)

    def run(self):
        for epoch in range(1, self.max_epoch+1):
            args.logger.info(f'Starting epoch {epoch}')
            self.train()
            self.save_ckpt(epoch)
            self.save_last_ckpt()
        self.save_log()

def main(args):
    Dataset = DataModule(args=args)
    Trainer = TrainModule(dataset=Dataset,
                      criterion=nn.CrossEntropyLoss(),
                      max_epoch=args.epoch,
                      mode=args.mode,
                      ckpt_name='',
                      args=args)
    Trainer.run()

def parser():
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('--epoch',
                        type=int,
                        default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help='lr')
    parser.add_argument('--mode',
                        type=str,
                        default='base')
    parser.add_argument('--expname',
                        default='EXP1',
                        help='name of exp')
    parser.add_argument('--base_label',
                        default='BASE1',
                        help='label for the base dataset, should be defined in Dataset.py/Config')
    parser.add_argument('--test_label',
                        default='TEST1',
                        help='label for the test dataset, should be defined in Dataset.py/Config')
    parser.add_argument('--pure_student',
                        type=bool,
                        default=False,
                        help='whether to train purely with the student model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser()
    if args.pure_student == False:
        if not os.path.exists(f'models/{args.expname}/{args.base_label}'):
            os.makedirs(f'models/{args.expname}/{args.base_label}')
        logger = log_creater(f'models/{args.expname}/{args.base_label}/log.log')
    else:
        if not os.path.exists(f'models/{args.expname}/{args.base_label}_pure_student'):
            os.makedirs(f'models/{args.expname}/{args.base_label}_pure_student')
        logger = log_creater(f'models/{args.expname}/{args.base_label}_pure_student/log.log')
    args.logger = logger
    args.logger.info(args)
    main(args)




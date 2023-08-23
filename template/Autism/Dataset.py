import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
import os
from torch.utils.data import Dataset
import pandas as pd

# BASE: 用于训练模型的训练集
# CAL: 用于评估memberhsip的校准数据集
# CALTEST: 校准数据集中的测试部分
# TEST: 模型训练过程中的测试集
# QO: defined query dataset overlapped with base training dataset
# QM: defined query dataset, k=来自训练集的百分比
# QN: query dataset没有和train overlap
# KD: 用于知识蒸馏的训练集，k=原始训练集的百分比, KD不能和query有overlap
# QF: query dataset for forget

CONFIG = {
'BASE1': {
    'BASE' : list(range(0,500))
},
'TEST1': {
    'TEST' : list(range(500,600))
},
'CAL_100': {
    'CAL' : list(range(500,600))
},
'CALTEST1': {
    'TEST' : list(range(600,700))
},
'QO_300': {
    'QUERY': list(range(0,400)),
    'QUERY_MEMBER': [1 for i in range(300)]
    },
'QNO_300': {
    'QUERY': list(range(700,1000)),
    'QUERY_MEMBER': [0 for i in range(300)]
    },
'QF_50':{
    'QUERY': list(range(450,500)),
    'QUERY_MEMBER': [1 for i in range(50)]
},
'QF_100':{
    'QUERY': list(range(400,500)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QS':{
    'QUERY': None,
    'QUERY_MEMBER': None
},
'KD0.25': {
    'BASE': list(range(0,125))
},
'KD0.5': {
    'BASE': list(range(0,250))
},
'KD0.75': {
    'BASE': list(range(0,375))
},
}

class AutismDataset(Dataset):

    def __init__(self):
        self.df = pd.read_csv('/home/zhouj0d/Science/PID24.AFB/src/template/Autism/final.csv')
        self.X = np.array(self.df.iloc[:, :-1])
        self.Y = np.array(self.df.iloc[:, -1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        return np.float32(self.X[index]), self.Y[index]


class AutismSimulateDataset(Dataset):

    def __init__(self):
        self.df = pd.read_csv('/home/zhouj0d/Science/PID24.AFB/src/template/Autism/final_simulated_10t_100f.csv')
        self.X = np.array(self.df.iloc[:, :-1])
        self.Y = np.array(self.df.iloc[:, -1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        return np.float32(self.X[index]), self.Y[index]

class DataModule():
    def __init__(self, batch_size=None, dir='.', num_workers=1, args=None):
        self.dir = dir
        self.data_dir = os.path.join(self.dir, 'data')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 2
        self.mode = args.mode
        self.args = args
        self.setup()

    def setup(self):
        if self.mode == 'base':
            trainset = AutismDataset()
            self.train_set = torch.utils.data.Subset(trainset, CONFIG[self.args.base_label]['BASE'])
            self.test_set = torch.utils.data.Subset(trainset, CONFIG[self.args.test_label]['TEST'])
            print(f'mode: {self.mode}, train: {len(self.train_set)}, test: {len(self.test_set)}')

        elif self.mode == 'cal':
            trainset = AutismDataset()
            self.train_set = torch.utils.data.Subset(trainset, CONFIG[self.args.cal_label]['CAL'])
            self.test_set = torch.utils.data.Subset(trainset, CONFIG[self.args.cal_test_label]['TEST'])
            print(f'mode: {self.mode}, train: {len(self.train_set)}, test: {len(self.test_set)}')

        elif self.mode == 'query':
            if 'QS' not in self.args.query_label:
                trainset = AutismDataset()
                self.test_set = torch.utils.data.Subset(trainset, CONFIG[self.args.query_label]['QUERY'])
                print(f'mode: {self.mode}, test: {len(self.test_set)}')
            else:
                trainset = AutismSimulateDataset()
                self.test_set = trainset
                print(f'mode: {self.mode}, test: {len(self.test_set)}')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

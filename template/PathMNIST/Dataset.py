import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
import os
import medmnist
from tqdm import tqdm

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
    'BASE' : list(range(0,10000))
},
'TEST1': {
    'TEST' : list(range(30000,35000))
},
'CAL_100': {
    'CAL' : list(range(40000,40100))
},
'CAL_1000': {
    'CAL' : list(range(40000,41000))
},
'CAL_2000': {
    'CAL' : list(range(40000,42000))
},
'CAL_5000': {
    'CAL' : list(range(40000,45000))
},
'CALTEST1': {
    'TEST' : list(range(46000,47000))
},
'QO_5000': {
    'QUERY': list(range(0,5000)),
    'QUERY_MEMBER': [1 for i in range(5000)]
    },
'QNO_5000': {
    'QUERY': list(range(50000,55000)),
    'QUERY_MEMBER': [0 for i in range(5000)]
    },
'QF_100':{
    'QUERY': list(range(9900,10000)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QF_1000':{
    'QUERY': list(range(9000,10000)),
    'QUERY_MEMBER': [1 for i in range(1000)]
},
'KD0.25': {
    'BASE': list(range(0,2500))
},
'KD0.5': {
    'BASE': list(range(0,5000))
},
'KD0.75': {
    'BASE': list(range(0,7500))
},
}

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class DataModule():
    def __init__(self, batch_size=32, dir='.', num_workers=1, args=None):
        self.dir = dir
        self.data_dir = os.path.join(self.dir, 'data')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 9
        self.mode = args.mode
        self.args = args
        self.download = True
        self.setup()

    def setup(self):
        if self.mode == 'base':
            trainset = medmnist.dataset.PathMNIST(root=self.dir, split='train', transform=transform_mnist, download=self.download)
            self.train_set = torch.utils.data.Subset(trainset, CONFIG[self.args.base_label]['BASE'])
            self.test_set = torch.utils.data.Subset(trainset, CONFIG[self.args.test_label]['TEST'])
            print(f'mode: {self.mode}, train: {len(self.train_set)}, test: {len(self.test_set)}')

        elif self.mode == 'cal':
            trainset = medmnist.dataset.PathMNIST(root=self.dir, split='train', transform=transform_mnist, download=self.download)
            self.train_set = torch.utils.data.Subset(trainset, CONFIG[self.args.cal_label]['CAL'])
            self.test_set = torch.utils.data.Subset(trainset, CONFIG[self.args.cal_test_label]['TEST'])
            print(f'mode: {self.mode}, train: {len(self.train_set)}, test: {len(self.test_set)}')

        elif self.mode == 'query':
            trainset = medmnist.dataset.PathMNIST(root=self.dir,split='train', transform=transform_mnist, download=self.download)
            self.test_set = torch.utils.data.Subset(trainset, CONFIG[self.args.query_label]['QUERY'])
            print(f'mode: {self.mode}, test: {len(self.test_set)}')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
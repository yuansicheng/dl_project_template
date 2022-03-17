#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-01-31

import torch
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
import os
import sys
import argparse
import logging

from copy import deepcopy

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')


class TrainController(ABC):
    def __init__(self,
                 data_loader=None,
                 models={},
                 lr=1e-4,
                 lr_decay=0,
                 epoch=100,
                 base_dir=None,
                 criterion=None,
                 loss_curve_log_y=False, 
                 use_tqdm=True, 
                 ckpt_path=None ) -> None:
        super().__init__()

        self.gpu = True if torch.cuda.is_available() else False

        assert data_loader
        self.data_loader = data_loader

        self.batch_size = self.data_loader.batch_size

        self.lr = lr
        self.lr_decay = lr_decay
        self.epoch = epoch

        # a dict, compatible with GAN
        assert models
        self.models = models
        self.optimizers = {}
        self.lr_schedulers = {}
        for k, v in self.models.items():
            if self.gpu:
                v.cuda()
            if ckpt_path:
                logging.info('Loading ckpt of {}'.format(k))
                v = torch.load(os.path.join(ckpt_path, '{}.ckpt'.format(k)))
            else:
                v.apply(self.weightInit)
            self.optimizers[k] = optim.SGD(v.parameters(), lr=self.lr)
            if self.lr_decay:
                self.lr_schedulers[k] = optim.lr_scheduler.StepLR(self.optimizers[k], lr_decay,gamma=0.5, last_epoch=-1, verbose=True)
            # summary(v, batch_size=self.batch_size)

        assert base_dir
        self.base_dir = base_dir
        self.img_dir = os.path.join(base_dir, 'img')
        os.mkdir(self.img_dir)

        self.criterion = criterion
        self.loss_curve_log_y = loss_curve_log_y
        self.use_tqdm = use_tqdm

        self.loss = pd.DataFrame()

        self.nepoch = 0
        self.nbatch = 0

    def weightInit(self, m):
        class_name = m.__class__.__name__
        if class_name.find('Linear') != -1 or \
           class_name.find('Conv') != -1:
            m.weight.data.normal_(-0.01, 0.01)

    def drawLoss(self):
        loss = deepcopy(self.loss)
        loss.to_csv(os.path.join(self.img_dir, 'loss.csv'))
        if not loss.shape[0]:
            return
        loss = loss.groupby('epoch').mean()
        plt.cla()
        plt.figure(figsize=(12, 4), dpi=256)
        plt.grid()
        plt.title('loss')
        plt.xlabel('epoch')
        if self.loss_curve_log_y:
            plt.yscale('log')
        for c in loss.columns:
            if 'train' in c:
                marker = 'o'
            elif 'validation' in c:
                marker = '+'
            else:
                continue
            plt.scatter(x=loss.index, y=loss[c],
                        marker=marker, alpha=0.7, label=c, zorder=1000)
        plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, 'loss.png'))
        plt.close()

    def saveModels(self):
        for k, v in self.models.items():
            torch.save(v, os.path.join(self.base_dir, '{}.ckpt'.format(k)))

    def train(self):
        for ie in range(self.epoch):
            self.nepoch = ie
            if not self.use_tqdm:
                logging.info('epoch: {}'.format(ie))
            for b in tqdm(range(self.data_loader.max_n_batch),
                          desc='epoch {}'.format(ie),
                          unit='batch',
                          disable=not self.use_tqdm):
                self.loss.loc[self.nbatch, 'epoch'] = ie
                # train
                if b < self.data_loader.max_n_batch*(1-self.data_loader.validation_ratio):
                    self.train_data = self.data_loader.getBatchTrainData()
                    self.trainOneBatch()

                # validation
                else:
                    self.validation_data = self.data_loader.getBatchValidationData()
                    self.validateOneBatch()

                self.nbatch += 1

            self.drawLoss()
            self.saveModels()
            self.data_loader.reset()
            self.afterEpoch()

            for v in self.lr_schedulers.values():
                v.step()

    @abstractmethod
    def trainOneBatch(self):
        pass

    @abstractmethod
    def validateOneBatch(self):
        pass

    def afterEpoch(self):
        pass

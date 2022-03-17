#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-01-30 

import os, sys, argparse, logging

import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

class DataLoader(ABC):
    def __init__(self, 
        h5_file_list = [],
        batch_size=128, 
        max_n_batch = 50, 
        validation_ratio=0.2) -> None:
        super().__init__()

        assert h5_file_list
        self.h5_file_list = h5_file_list
        self.h5_file_index = 0
        self.max_n_batch = max_n_batch
        self.train_batch_index = 0
        self.test_batch_index = 0
        self.batch_size = batch_size
        self.validation_ratio = validation_ratio

        self.train_data = {'input': None, 'label':None}
        self.validation_data = {'input': None, 'label':None}

        self.train_data_len = 0
        self.validation_data_len = 0



    @abstractmethod
    def loadOneFile(self, h5_file, *args, **kwargs):
        pass

    def addOneFileData(self):
        data = self.loadOneFile(self.h5_file_list[self.h5_file_index])
        index = int(data['input'].shape[0] * self.validation_ratio)

        for key in ['input', 'label']:
            if self.train_data_len:
                self.train_data[key] = np.concatenate([self.train_data[key], data[key][index:]])
            else:
                self.train_data[key] = data[key][index:].copy()
            if self.validation_data_len:
                self.validation_data[key] = np.concatenate([self.validation_data[key], data[key][: index]])
            else:
                self.validation_data[key] = data[key][:index].copy()

        self.h5_file_index += 1

        self.train_data_len = self.train_data['input'].shape[0]
        self.validation_data_len = self.validation_data['input'].shape[0]

    def reset(self):
        self.h5_file_index = 0  
        self.train_data = {'input': None, 'label':None}
        self.validation_data = {'input': None, 'label':None}

        self.train_data_len = 0
        self.validation_data_len = 0   

        self.train_batch_index = 0
        self.test_batch_index = 0

    def getBatchTrainData(self):
        if self.train_data_len < self.batch_size:
            if self.h5_file_index == len(self.h5_file_list):
                logging.fatal('Data not enough!')
            self.addOneFileData()
        data = {k:v[:self.batch_size] for k,v in self.train_data.items()}
        self.train_data = {k:v[self.batch_size:] for k,v in self.train_data.items()}
        self.train_batch_index += 1
        self.train_data_len = self.train_data['input'].shape[0]
        return data

    def getBatchValidationData(self):
        data = {k:v[:self.batch_size] for k,v in self.validation_data.items()}
        self.validation_data = {k:v[self.batch_size:] for k,v in self.validation_data.items()}
        self.validation_data_len = self.validation_data['input'].shape[0]
        self.test_batch_index += 1
        return data

    def getShape(self):
        data = self.getBatchTrainData()
        shape = {k:v.shape[1:] for k,v in data.items()}
        self.reset()
        return shape
        



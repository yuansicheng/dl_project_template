#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-01 

import os, sys, argparse, logging

import h5py
import numpy as np

from base_classes.data_loader import DataLoader

class MyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def loadOneFile(self, h5_file, *args, **kwargs):
        logging.debug('Loading {}'.format(h5_file))
        hf = h5py.File(h5_file, 'r')
        input = hf['leading_shower_5x5'][:][:,np.newaxis,:,:]
        label = np.concatenate((
            hf['momentum'][:][:, np.newaxis], 
            hf['theta'][:][:, np.newaxis], 
        ), axis=-1)
        return {
            'input': input, 
            'label': label, 
        }
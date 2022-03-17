#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-01-30

import os
import sys
import argparse
import logging

from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
import h5py
from tqdm import tqdm

from base_classes.utils import openRoot


class RootExtractor(ABC):
    def __init__(self,
                 root_file_list=[],
                 edm=False,
                 step='',
                 tree_name='',
                 h5_path='',
                 event_per_file=10000,
                 h5_file_num=10,
                 **kwargs) -> None:
        super().__init__()

        self.root_file_list = root_file_list

        self.edm = edm
        self.step = step
        self.tree_name = tree_name

        self.h5_path = h5_path
        self.event_per_file = int(event_per_file)
        self.h5_file_num = h5_file_num

        # format of self.data
        # {
        #   key1: [np.array, np.array, ...],
        #   key2: ...
        # }
        self.data = defaultdict(list)

        self.root_file_index = 0

        self.f = None
        self.tree = None

    @abstractmethod
    def extractOneEvent(self):
        pass

    def saveOneH5(self, i):
        if not self.data:
            return False
        h5_file_name = os.path.join(
            self.h5_path, '{}.h5'.format(i))
        hf = h5py.File(h5_file_name, 'w')
        for key, value in self.data.items():
            if len(value) < self.event_per_file:
                return False
            hf.create_dataset(key, data=np.array(value[:self.event_per_file]))
            self.data[key] = value[self.event_per_file:]
        hf.close()
        logging.info('saved hdf5: ' + h5_file_name)
        return True

    def extractOneFile(self):
        assert self.root_file_index < len(self.root_file_list)
        self.f, self.tree, entries = openRoot(self.root_file_list[self.root_file_index], edm=self.edm, step=self.step, tree_name=self.tree_name)
        for ie in tqdm(range(entries),
                       desc='extract root file',
                       unit='events'):
            if self.data and len(list(self.data.values())[0]) > self.h5_file_num * self.event_per_file:
                break
            self.tree.GetEntry(ie)
            # append event data to self.data
            self.extractOneEvent()
        self.root_file_index += 1
        return

    def extract(self):
        for i in tqdm(range(self.h5_file_num),
                      desc='save hdf5',
                      unit='files'):
            while not self.saveOneH5(i):
                self.extractOneFile()
            


#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-22 

import os, sys, argparse, logging

import os
import sys
import argparse
import logging

from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from base_classes.utils import openRoot


class RootPainter(ABC):
    def __init__(self,
                 root_file='',
                 edm=False,
                 step='',
                 tree_name='',
                 max_n=100, 
                 img_dir = '', 
                 **kwargs) -> None:
        super().__init__()

        self.root_file = root_file
        self.edm = edm
        self.step = step
        self.tree_name = tree_name
        self.max_n = max_n
        self.img_dir = img_dir

        self.data = defaultdict(list)

        self.f = None
        self.tree = None

    @abstractmethod
    def extractOneEvent(self):
        pass

    @abstractmethod
    def paint(self):
        pass

    def list2Numpy(self):
        for key, value in self.data.items():
            try:
                self.data[key] = np.array(value)
            except Exception as e:
                logging.error(e)

    def extract(self):
        self.f, self.tree, entries = openRoot(self.root_file, edm=self.edm, step=self.step, tree_name=self.tree_name)
        for ie in tqdm(range(int(min(entries, self.max_n))),
                       desc='extracting root',
                       unit='events'):
            self.tree.GetEntry(ie)
            # append event data to self.data
            self.extractOneEvent()

    def run(self):
        '''
        user api
        '''
        self.extract()
        self.list2Numpy()
        self.paint()
            

    

#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-22 

import os, sys, argparse, logging
from glob import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from my_root_extractor import MyRootExtractor


####################################################
####################################################
# ARGS
debug = True

root_file_list=glob(r'../data/root/*.root')
edm=False
step=''
tree_name='EmcInfo'
h5_path='../data/h5'
event_per_file=1e4
h5_file_num=5

emc_info_file = '../data/emc_info.csv'

####################################################
####################################################

# set loglevel
if debug:
    loglevel = logging.DEBUG
else:
    loglevel = logging.INFO
logging.basicConfig(level=loglevel, format="%(asctime)s-%(filename)s[line:%(lineno)d]-%(funcName)s-%(levelname)s : %(message)s")

# check h5_path
if not os.path.exists(h5_path):
    os.makedirs(h5_path)

my_root_extractor = MyRootExtractor(root_file_list=root_file_list,
                                    edm=edm,
                                    step=step,
                                    tree_name=tree_name,
                                    h5_path=h5_path, 
                                    event_per_file=event_per_file, 
                                    h5_file_num=h5_file_num, 
                                    emc_info_file=emc_info_file,)

my_root_extractor.extract()


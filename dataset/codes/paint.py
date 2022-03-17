#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-22 

import os, sys, argparse, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from my_root_painter import MyRootPainter

####################################################
####################################################
# ARGS
debug = False

root_file = '../data/root/0.root'
edm = False
step = ''
tree_name = 'EmcInfo'
max_n = 1e4
img_dir = '../img'

emc_info_file = '../data/emc_info.csv'
####################################################
####################################################

# set loglevel
if debug:
    loglevel = logging.DEBUG
else:
    loglevel = logging.INFO
logging.basicConfig(level=loglevel, format="%(asctime)s-%(filename)s[line:%(lineno)d]-%(funcName)s-%(levelname)s : %(message)s")

# check img_dir
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

my_root_painter = MyRootPainter(root_file=root_file,
                                edm=edm,
                                step=step,
                                tree_name=tree_name,
                                max_n=max_n, 
                                img_dir=img_dir,
                                emc_info_file=emc_info_file)

my_root_painter.run()


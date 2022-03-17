#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-11 

import os, sys, argparse, logging
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--h5_file', type=str,  default='')
parser.add_argument('--n', type=int,  default=0)
args = parser.parse_args()

h5_file = 'dataset/data/merge.h5'
n = 16384
if not args.h5_file:
    args.h5_file = h5_file
if not args.n:
    args.n = n

assert args.h5_file
assert args.n

data = {}
hf = h5py.File(args.h5_file, 'r')

for k in hf.keys():
    data[k] = hf[k][:][:args.n]

new_file = os.path.join(os.path.dirname(args.h5_file), args.h5_file.split('/')[-1][:-3]+'_cut.h5')
hf = h5py.File(new_file, 'w')
for key, value in data.items():
    hf.create_dataset(key, data=value)
hf.close()
print('writeHDF5: {}'.format(new_file))

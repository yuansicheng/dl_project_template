#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-01

import os
import sys
import argparse
import logging
from datetime import datetime
import shutil
from glob import glob

from torch import nn

from my_data_loader import *
from my_model import *
from my_train_controller import *

this_file = os.path.abspath(__file__)
this_path = os.path.dirname(this_file)

# arg for sbatch
parser = argparse.ArgumentParser()
parser.add_argument('--sbatch_job', type=bool,  default=False)
parser.add_argument('--timestamp', type=str,  default='')
args = parser.parse_args()

# base_dir
timestamp = args.timestamp if args.timestamp else datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
if args.sbatch_job:
    timestamp = 'sbatch_' + timestamp
elif not args.timestamp:
    timestamp = 'local_' + timestamp
base_dir = os.path.join(this_path, '../jobs', timestamp)

####################################################
####################################################
# GLOBAL ARGS

# debug = True
debug = False

# DEEP LEARNING ARGS
# for data_loader
h5_file_path = os.path.join(this_path, '../dataset/data/h5')
h5_file_list = glob('{}/*.h5'.format(h5_file_path))
batch_size = 128
max_n_batch = 300
validation_ratio = 0.2

# for model

# for train_controller
lr = 1e-3
lr_decay = 50
epochs = 1000
criterion = nn.MSELoss()
loss_curve_log_y = False
use_tqdm = True
# ckpt_path = os.path.join(base_dir, '..', 'local_2022-02-11-15-14-53')
ckpt_path = None

noise_dim = 2

####################################################
####################################################

# set loglevel
if debug:
    loglevel = logging.DEBUG
else:
    loglevel = logging.INFO
logging.basicConfig(level=loglevel, format="%(asctime)s-%(filename)s[line:%(lineno)d]-%(funcName)s-%(levelname)s : %(message)s")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)
os.chdir(base_dir)


if not args.timestamp:
    # copy scripts folder as checkpoint
    logging.info('Copying scripts')
    shutil.copytree(this_path, os.path.join(base_dir, 'scripts'))

if args.sbatch_job:
    job_file = os.path.join(base_dir, 'scripts/job.sh')
    with open(job_file, 'a') as f:
        f.write('\n\t--timestamp {}'.format(timestamp))
    os.system('chmod 755 {}'.format(job_file))
    os.system('sbatch {}'.format(job_file))
    sys.exit(1)


# init data_loader
logging.info('Initializing data_loader')
data_loader = MyDataLoader(
    h5_file_list=h5_file_list,
    batch_size=batch_size,
    max_n_batch=max_n_batch,
    validation_ratio=validation_ratio
)

# init models
logging.info('Initializing models')
models = {
    'G': G(noise_dim=noise_dim), 
    'D': D(), 
}

# init train_controller
logging.info('Initializing train_controller')
train_controller = MyTrainController(
    data_loader=data_loader,
    models=models,
    lr=lr,
    lr_decay=lr_decay,
    epoch=epochs,
    base_dir=base_dir, 
    criterion=criterion,
    loss_curve_log_y=loss_curve_log_y, 
    use_tqdm=use_tqdm, 
    ckpt_path=ckpt_path, 
    noise_dim=noise_dim, 
)


train_controller.train()


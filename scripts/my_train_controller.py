#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-01 

import os, sys, argparse, logging

import torch
from torch import Tensor

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.set_loglevel('info')

from base_classes.train_controller import TrainController

class MyTrainController(TrainController):
    def __init__(self, *args, noise_dim=1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_dim = noise_dim

    def addNoise(self, data, noise_dim):
        # logging.debug(data.shape)
        return np.concatenate((data, np.random.normal(size=(data.shape[0],noise_dim))), axis=-1)

    def trainOneBatch(self):
        g_input = Tensor(self.addNoise(self.train_data['label'], self.noise_dim))
        real_img = Tensor(self.train_data['input'])
        real_label = Tensor(np.ones((self.batch_size, 1)))
        fake_label = Tensor(np.zeros((self.batch_size, 1)))

        if self.gpu:
            torch.cuda.empty_cache()
            g_input = g_input.cuda()
            real_img = real_img.cuda()
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()
        ##############################################
        # train G
        ##############################################
        fake_img = self.models['G'](g_input)
        fake_out = self.models['D'](fake_img)

        g_loss = self.criterion(fake_out, real_label)

        self.optimizers['G'].zero_grad()
        g_loss.backward()
        self.optimizers['G'].step()

        self.loss.loc[self.nbatch, 'train_g_loss'] = g_loss.cpu().detach().numpy()
        logging.debug('g_loss: {}'.format(self.loss.loc[self.nbatch, 'train_g_loss']))

        ##############################################
        # train D
        ##############################################
        real_out = self.models['D'](real_img)
        d_loss_real = self.criterion(real_out, real_label)
        fake_img = self.models['G'](g_input)
        fake_out = self.models['D'](fake_img)
        d_loss_fake = self.criterion(fake_out, fake_label)

        d_loss = (d_loss_real + d_loss_fake) / 2

        self.optimizers['D'].zero_grad()
        d_loss.backward()
        self.optimizers['D'].step()

        self.loss.loc[self.nbatch, 'train_d_loss'] = d_loss.cpu().detach().numpy()
        logging.debug('d_loss: {}'.format(self.loss.loc[self.nbatch, 'train_d_loss']))
        self.loss.loc[self.nbatch, 'train_d_loss_real'] = d_loss_real.cpu().detach().numpy()
        logging.debug('d_loss_real: {}'.format(self.loss.loc[self.nbatch, 'train_d_loss_real']))
        self.loss.loc[self.nbatch, 'train_d_loss_fake'] = d_loss_fake.cpu().detach().numpy()
        logging.debug('d_loss_fake: {}'.format(self.loss.loc[self.nbatch, 'train_d_loss_fake']))


    def validateOneBatch(self):
        pass
        

    def afterEpoch(self):
        self.drawImg(20)
        

    def drawImg(self, n):
        g_input = Tensor(self.addNoise(self.validation_data['label'], self.noise_dim))
        real_img = self.validation_data['input']
        if self.gpu:
            torch.cuda.empty_cache()
            g_input = g_input.cuda() 
        with torch.no_grad():
            fake_img = self.models['G'](g_input)
        
        fake_img = fake_img.cpu().detach().numpy()

        np.place(real_img, real_img<1e-5, np.nan)
        np.place(fake_img, fake_img<1e-5, np.nan)

        logging.debug(real_img.shape)
        logging.debug(fake_img.shape)

        with PdfPages(os.path.join(self.img_dir, '{}_{}.pdf'.format('fake_img', self.nepoch))) as pdf:
            for i in range(min(n, self.batch_size)):
                plt.cla()
                plt.figure(figsize=(8,6), dpi=256)
                plt.imshow(fake_img[i].T, zorder=10)
                plt.xlabel('ntheta')
                plt.ylabel('nphi')
                plt.colorbar()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        if not os.path.exists(os.path.join(self.img_dir, 'real_img.pdf')):
            with PdfPages(os.path.join(self.img_dir, 'real_img.pdf')) as pdf:
                for i in range(self.batch_size):
                    plt.cla()
                    plt.figure(figsize=(8,6), dpi=256)
                    plt.imshow(real_img[i].T, zorder=10)
                    plt.xlabel('ntheta')
                    plt.ylabel('nphi')
                    plt.colorbar()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()



        

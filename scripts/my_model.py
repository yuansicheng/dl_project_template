#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-01 

import os, sys, argparse, logging
from torch import nn

from base_classes.model import *


class G(Model):
    def __init__(self, *args, noise_dim=1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_dim = noise_dim
        self.buildModel()

    def buildModel(self):
        self.fc = nn.Sequential()
        addLinearBlock(self.fc, self.noise_dim+2, 128)
        addLinearBlock(self.fc, 128, 512)
        addLinearBlock(self.fc, 512, 1024)
        addLinearBlock(self.fc, 1024, 1024)

        self.conv = nn.Sequential()
        addConv2dBlock(self.conv, 16, 16, 2, padding=0)
        addConv2dBlock(self.conv, 16, 16, 2, padding=0)
        addConv2dBlock(self.conv, 16, 1, 2, padding=0)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16, 8, 8)
        x = self.conv(x)
        return x

class D(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.buildModel()

    def buildModel(self):
        self.conv = nn.Sequential()
        addConv2dBlock(self.conv, 1, 16, 2, padding=1)
        addConv2dBlock(self.conv, 16, 16, 2, padding=1)
        addConv2dBlock(self.conv, 16, 16, 2, padding=1)

        self.fc = nn.Sequential()
        addLinearBlock(self.fc, 1024, 1024)
        addLinearBlock(self.fc, 1024, 256)
        addLinearBlock(self.fc, 256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-01-31 

import os, sys, argparse, logging

from abc import ABC, abstractmethod
from turtle import forward

import torch.nn as nn

def addLinearBlock(s, input_dim, output_dim):
    n = len(s)
    s.add_module(str(n), nn.Linear(input_dim, output_dim))
    s.add_module(str(n+1), nn.BatchNorm1d(output_dim))
    s.add_module(str(n+2), nn.Dropout(0.2))
    s.add_module(str(n+3), nn.LeakyReLU(0.2, inplace=True))

def addConv2dBlock(s, in_channels, out_channels, kernel_size, stride=1, padding=1):
    n = len(s)
    s.add_module(str(n), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    s.add_module(str(n+1), nn.BatchNorm2d(out_channels))
    s.add_module(str(n+2), nn.Dropout(0.2))
    s.add_module(str(n+3), nn.LeakyReLU(0.2, inplace=True))

def addConv2dTransposeBlock(s, in_channels, out_channels, kernel_size, stride=1, padding=1):
    n = len(s)
    s.add_module(str(n), nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    s.add_module(str(n+1), nn.BatchNorm2d(out_channels))
    s.add_module(str(n+2), nn.Dropout(0.2))
    s.add_module(str(n+3), nn.LeakyReLU(0.2, inplace=True))

class Model(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def buildModel(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, x):
        pass



#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2021-12-18 

import os, sys, argparse, logging

import ROOT as rt
# import h5py
# import pandas as pd

def openJunoEdmRoot(edm_rootfile, step='Sim') -> tuple:
    '''
    Open juno edm root file,
    note: must use official juno environment!
    '''
    assert edm_rootfile and isinstance(step,str) and step.lower() in ('sim', 'calib', 'rec')
    logging.info('Opening: {}'.format(edm_rootfile))
    step = step[0].upper() + step[1:].lower()

    rt.gSystem.Load("libEDMUtil")
    rt.gSystem.Load("libSimEventV2")
    rt.gSystem.Load("libCalibEvent")
    rt.gSystem.Load("libRecEvent")

    # concat tree_name
    tree_name = 'Event/{}/{}Event'.format(step, step)

    f = rt.TFile.Open(edm_rootfile)
    tree = f.Get(tree_name)
    entries = tree.GetEntries()

    return  f, tree, entries

def openNormalRoot(root_file, tree_name='') -> tuple:
    '''
    Open normal root file.
    '''
    assert root_file and tree_name
    logging.info('Opening: {}'.format(root_file))

    f = rt.TFile(root_file)
    tree = f.Get(tree_name)
    entries = tree.GetEntriesFast()
    return f, tree, entries

def openRoot(root_file, edm=False, **kwargs):
    if edm:
        assert 'step' in kwargs
        return openJunoEdmRoot(root_file, step=kwargs['step'])
    else:
        assert 'tree_name' in kwargs
        return openNormalRoot(root_file, tree_name=kwargs['tree_name'])






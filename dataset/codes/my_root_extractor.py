#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-22 

import os, sys, argparse, logging
import pandas as pd
import numpy as np

from base_classes.root_extractor import RootExtractor

class MyRootExtractor(RootExtractor):
    def __init__(self, *args,emc_info_file='',  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setEmcInfo(emc_info_file)


    def setEmcInfo(self, emc_info_file):
        self.emc_info = pd.read_csv(emc_info_file)
        self.emc_info = self.emc_info[self.emc_info['npart']==1]
        self.emc_info.index = self.emc_info['id']
        self.emc_shape = self.emc_info[['ntheta', 'nphi']].max() + 1


    def extractOneEvent(self):
        # leading shower in barrel
        leading_shower_id = getattr(self.tree, 'leading_shower_id')
        if leading_shower_id not in self.emc_info.index:
            return

        self.data['momentum'].append(getattr(self.tree, 'momentum')) 
        self.data['theta'].append(getattr(self.tree, 'theta'))
        self.data['phi'].append(getattr(self.tree, 'phi'))

        # leading shower 5x5
        leading_shower_coordinate = self.emc_info.loc[leading_shower_id]
        leading_shower_5x5 = np.zeros((5, 5))
        all_hit_cell_id = list(getattr(self.tree, 'all_hit_cell_id'))
        all_hit_energy = list(getattr(self.tree, 'all_hit_energy'))
        all_hit_dict = dict(zip(all_hit_cell_id, all_hit_energy))
        leading_shower_hit_cell_id_5x5 = list(getattr(self.tree, 'leading_shower_hit_cell_id_5x5'))
        for i in range(len(leading_shower_hit_cell_id_5x5)):
            if leading_shower_hit_cell_id_5x5[i] not in self.emc_info.index:
                continue
            tmp = self.emc_info.loc[leading_shower_hit_cell_id_5x5[i]]
            leading_shower_5x5[tmp['ntheta']-leading_shower_coordinate['ntheta']+2, (tmp['nphi']-leading_shower_coordinate['nphi']+2+self.emc_shape['nphi'])%self.emc_shape['nphi']] = all_hit_dict[leading_shower_hit_cell_id_5x5[i]]
        self.data['leading_shower_5x5'].append(leading_shower_5x5)



        
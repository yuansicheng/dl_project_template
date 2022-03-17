#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-22 

import os, sys, argparse, logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
plt.set_loglevel("info") 

from tqdm import tqdm

from base_classes.root_painter import RootPainter

class MyRootPainter(RootPainter):
    def __init__(self, *args, emc_info_file='', **kwargs) -> None:
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

        # emc hits
        hit_img = np.full((self.emc_shape['ntheta'], self.emc_shape['nphi']), np.nan)
        all_hit_cell_id = list(getattr(self.tree, 'all_hit_cell_id'))
        all_hit_energy = list(getattr(self.tree, 'all_hit_energy'))
        for i in range(len(all_hit_cell_id)):
            try:
                tmp = self.emc_info.loc[all_hit_cell_id[i]]
                hit_img[tmp['ntheta'], tmp['nphi']] = all_hit_energy[i]
            except:
                pass
        self.data['hit_img'].append(hit_img)

        # leading shower id
        tmp = self.emc_info.loc[leading_shower_id]
        self.data['leading_shower_coordinate'].append([tmp['ntheta'], tmp['nphi']])



    def paint(self):
        # self.paintHist()
        self.paintHitImg(100)

        

    def paintHist(self):
        keys = ['momentum', 'theta', 'phi']
        df = pd.DataFrame({k:v for k,v in self.data.items() if k in keys})

        for key in df.columns:
            plt.cla()
            plt.figure(figsize=(4,3), dpi=256)
            ax = plt.axes()
            df[key].hist(ax=ax, bins=20, histtype='step')
            plt.grid()
            # plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
            plt.title(key)
            plt.tight_layout()
            plt.savefig(os.path.join(self.img_dir, '{}.png'.format(key)))
            plt.close()
            logging.info('Saved {}'.format(os.path.join(self.img_dir, '{}.png'.format(key))))

    def paintHitImg(self, n):
         with PdfPages(os.path.join(self.img_dir, 'hit_img.pdf')) as pdf:
            for i in tqdm(range(n),
                          desc='draw hit img',
                          unit='figs'):
                plt.cla()
                plt.figure(figsize=(7,12), dpi=256)
                ax = plt.axes()
                ax.add_patch(patches.Rectangle(self.data['leading_shower_coordinate'][i]-2.5, 5, 5, fill=False, linewidth=1, edgecolor='red', zorder=100))
                plt.imshow(self.data['hit_img'][i].T, zorder=10)
                plt.xlabel('ntheta')
                plt.ylabel('nphi')
                plt.colorbar()
                plt.grid()
                plt.title('momentum={}'.format(self.data['momentum'][i]))
                plt.tight_layout()
                pdf.savefig()
                plt.close()


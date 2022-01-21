import os
import json
import time
import random

import click
import cv2
import torch
import numpy as np
from PIL import Image

from deepliif.data import create_dataset, AlignedDataset, transform
from deepliif.models import inference, postprocess, compute_overlap, init_nets, DeepLIIFModel
from deepliif.util import allowed_file, Visualizer

import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='options for running a local visdom visualizer that consumes pickled snapshots')
    parser.add_argument('--pickle_dir',required=True,type=str,default='./pickle',help='directory where the pickled snapshots are stored')
    parser.add_argument('--n_proc',type=int,default=1,help='number of processes of this experiment, used to properly calculate the effective counter ratio')
    
    args = parser.parse_args()
    pickle_dir = args.pickle_dir
    n_proc = args.n_proc
    
    path_init = os.path.join(pickle_dir,'opt.pickle')
    print(f'waiting for initialization signal from {path_init}')
    while not os.path.exists(path_init):
        time.sleep(1)
        
    params_opt = pickle.load(open(path_init,'rb'))
    params_opt['remote'] = False
    visualizer = Visualizer(**params_opt)   # create a visualizer that display/save images and plots

    paths_plot = {'display_current_results':os.path.join(pickle_dir,'display_current_results.pickle'),
                  'plot_current_losses':os.path.join(pickle_dir,'plot_current_losses.pickle')}
    
    last_modified_time = {k:0 for k in paths_plot.keys()} # initialize time
    
    while True:
        for method, path_plot in paths_plot.items():
            try:
                last_modified_time_plot = os.path.getmtime(path_plot)
                if last_modified_time_plot > last_modified_time[method]:
                    params_plot = pickle.load(open(path_plot,'rb'))
                    params_plot['n_proc'] = n_proc
                    last_modified_time[method] = last_modified_time_plot
                    getattr(visualizer,method)(**params_plot)
                    print(f'{method} refreshed, last modified time {time.ctime(last_modified_time[method])}')
                else:
                    print(f'{method} not refreshed')
            except Exception as e:
                print(e)
        time.sleep(10)

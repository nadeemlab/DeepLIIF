"""General-purpose training script for multi-task image-to-image translation.

This script works for various models (with option '--model': e.g., DeepLIIF) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
"""
import time
from deepliif.options.train_options import TrainOptions
from deepliif.data import create_dataset
from deepliif.models import create_model
from deepliif.util.visualizer import Visualizer

import pickle
import os
import numpy as np
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
        
    opt = pickle.load(open(path_init,'rb'))
    opt.remote = False
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

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

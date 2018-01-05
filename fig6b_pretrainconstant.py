"""Recreates Figure 5C from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *

ms = 3000
n_iters = 10
pretunes = [1, 5, 10]

############## UNCOMMENT TO RUN NEW RESULTS #############
'''
args = get_args()
args['tuned'] = False
args['ms'] = ms
args['n_iters'] = n_iters
args['n_neurons'] = 100 
total_time = 0

## run networks in parallel
for i, n in enumerate(pretunes):
    args['pretune'] = n
    print('Running ' + str(n) + ' training sessions ')
    f_stim = ['storing/pretrain_constant_ntrains:' + str(n) +'.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run frac:' + str(n) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')
'''
dfs = list()
labels = list()

## add no previous tunings (constant model)
f_stim = 'storing/constant_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
dfs.append(stims[:ms])
labels.append(' n = 0')

### read in files 
for i, n in enumerate(pretunes):
    f_stim = 'storing/pretrain_constant_ntrains:' + str(n) + '.csv'  ## comment to plot new results 
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(' n = ' + str(n))
pa = plot_args(n_curves = len(dfs))
pa['labels'] = labels
pa['plt_errors'] = True
pa['yspots'] = [-.02, .55, .7, .85]
pa['xspot'] = 3000
pa['right_margins'] = True
pa['ylim'] = [-.1, 1.1]
pa['title'] = 'Constant Pre-Trained'
plot_pqs(dfs, pa)

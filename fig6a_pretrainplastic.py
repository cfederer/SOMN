"""Not same data as in arxiv paper but same methods as Figure 5A """
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
args['tuned'] = True 
args['ms'] = ms
args['n_iters'] = n_iters
total_time = 0

## run networks in parallel
for i, n in enumerate(pretunes):
    args['pretune'] = n
    print('Running ' + str(n) + ' training sessions ')
    f_stim = ['storing/pretrain_plastic_ntrains_:' + str(n) +'.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run frac:' + str(n) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')
'''

dfs = list()
labels = list()

## add no previous tunings 
f_stim = 'storing/plastic_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
dfs.append(stims[:ms])
labels.append(' n = 0')

### read in files 
for i, n in enumerate(pretunes):
    f_stim = 'storing/pretrain_plastic_ntrains_:' + str(n) + '.csv' ## comment out to plot new results 
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(' n = ' + str(n))
pa = plot_args(n_curves = 6)
pa['colors'] = pa['colors'][2:]
pa['labels'] = labels
pa['title'] = 'Plastic Pre-Trained'
pa['xspot'] = 3000
pa['yspots'] = [.7, .88, .96, 1.03]
pa['plt_errors'] = True
pa['text_size'] = pa['text_size'] - 2
pa['right_margins'] = True
plot_pqs_cutaxis(dfs, pa,[.9968, 1.0001], [0,.5])

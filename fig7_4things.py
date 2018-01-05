"""Recreates Figure 3A from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *


ms = 3000
n_iters = 10
n_stim = 4
############## UNCOMMENT TO RUN NEW RESULTS #############
'''
args_t = get_args()
args_t['ms'] = ms
args_t['n_iters'] = n_iters
args_t['tuned'] = True
args_t['n_stim'] = n_stim

args_u = get_args()
args_u['ms'] = ms
args_u['n_iters'] = n_iters
args_u['tuned'] = False 
args_u['n_stim'] = n_stim

### create files for storing stim values
t_stims = list()
u_stims = list()
for i in range(args_t['n_stim']):
    tstim = 'storing/plastic_stim_'+ str(i+1) +'.csv'
    t_stims.append(tstim)
    initialize_file(tstim, args_t)
    ustim = 'storing/constant_stim_'+ str(i+1) + '.csv'
    u_stims.append(ustim)
    initialize_file(ustim, args_u)
## run networks in parallel               
tt = PQ(args_t, stim_l = t_stims)
ut = PQ(args_u, stim_l = u_stims)

print('Time to run tuned ' + str(tt/60) + ' minutes')
print('Time to run untuned ' + str(ut/60) + ' minutes')
'''
## create lists for plots 
dfs = list()
tdfs = list()
tLabels = list()

for i in range(n_stim):
    ## open and read stim
    ts = pd.read_csv('storing/plastic_stim_'+ str(i+1) + '.csv') ##comment out to run new results
    dfs.append(pd.DataFrame.transpose(ts))
    tdfs.append(pd.DataFrame.transpose(ts))
    tLabels.append('Plastic Random Synapse Stim ' + str(i+1))
for i in range(n_stim): 
    us = pd.read_csv('storing/constant_stim_'+ str(i+1) + '.csv') ##comment out to run new results
    dfs.append(pd.DataFrame.transpose(us))
    

## plot Fig 3A
pa = plot_args(n_curves=2, rb=False)
pa['colors'] = ['m', 'b', 'g', 'c', 'm', 'b', 'g', 'c']
pa['text_colors'] = ['b', 'b', 'b', 'b', 'r', 'r', 'r', 'r']
pa['labels'] = ['Plastic Random Synapse Stims (1-4)', '', '', '', 'Constant Random Synapse Stims (1-4)', '', '', '']
pa['linestyle'] = ['-', '-', '-', '-', '--', '--', '--', '--']
pa['plt_errors'] = True
pa['xspot'] =  45
pa['yadds'] = [-.1, 0, 0, 0, .03, 0, 0, 0]
pa['text_size'] = 16
pa['xlim'] = [-1, 2999]
plot_pqs(dfs, pa)

## plot Fig 3B
pa['xspot'] = 250
pa['plt_errors'] = False
pa['text_colors'] = pa['colors']
pa['labels'] = ['Plastic Random Synapse Stim 1', 'Plastic Random Synapse Stim 4', 'Plastic Random Synapse Stim 2', 'Plastic Random Synapse Stim 3']
pa['yadds'] = [.00003, .00006, .00004, .00002]
pa['ylabel'] = ''
plot_pqs(tdfs, pa)

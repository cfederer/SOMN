"""Recreates Figure 3A from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NeuralNetwork import NN
from NetworkSimulation import Sim
from quantify_sims_multithreading import * 
import pandas as pd
import numpy as np
from arg_parsing import *
from plot_util import *


ms = 3000
n_iters = 100
n_stim = 4
############## UNCOMMENT TO RUN NEW RESULTS #############
'''
## add random str in to not over-write 
r = str(abs(np.round(np.random.randn(), 4)))

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
    tstim = 'storing/plastic_stim_'+ str(i+1) + '_' + r + '.csv'
    t_stims.append(tstim)
    initialize_file(tstim, args_t)
    ustim = 'storing/constant_stim_'+ str(i+1) + '_' + r + '.csv'
    u_stims.append(ustim)
    initialize_file(ustim, args_u)
                
tt = PQ(args_t, stim_l = t_stims)
ut = PQ(args_u, stim_l = u_stims)

## run networks in parallel
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
    #ts = pd.read_csv('storing/plastic_stim_'+ str(i+1) + '_' + r + '.csv') ##uncomment to run new results 
    dfs.append(pd.DataFrame.transpose(ts))
    tdfs.append(pd.DataFrame.transpose(ts))
    tLabels.append('Plastic Stim ' + str(i+1))
for i in range(n_stim): 
    us = pd.read_csv('storing/constant_stim_'+ str(i+1) + '.csv') ##comment out to run new results
    #us= pd.read_csv('storing/constant_stim_'+ str(i+1) + '_' + r + '.csv') ##uncomment to run new results 
    dfs.append(pd.DataFrame.transpose(us))
    

## plot Fig 3A
pa = plot_args()
pa['colors'] = ['orangered', 'blue', 'green', 'purple', 'orangered', 'blue', 'green', 'purple']
pa['error_colors'] = ['#ff9999','#8080ff', '#c6ecd7','#ccb3e6']
pa['labels'] = ['Plastic Stims (1-4)', '', '', '', 'Constant Stims (1-4)', '', '', '']
pa['linestyle'] = ['-', '-', '-', '-', '--', '--', '--', '--']
pa['linewidth'] = 3
pa['xspot'] =  500
pa['yadds'] = [-.1, 0, 0, 0, .03, 0, 0, 0]
plot_pqs(dfs, pa)

## plot Fig 3B
pa['ylim'] = [.99825, 1.00001]
pa['labels'] = tLabels
pa['yadds'] = [.00003, .00003, .00003, .00003]
pa['ylabel'] = '' 
plot_pqs(tdfs, pa)



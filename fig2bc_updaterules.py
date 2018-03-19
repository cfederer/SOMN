"""Recreates Figure 2BC from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *


ms = 3000
n_iters = 10 
############## UNCOMMENT TO RUN NEW RESULTS #############
'''
args_t = get_args()
args_t['tuned'] = True
args_t['ms'] = ms
args_t['n_iters'] = n_iters

args_u = get_args()
args_u['tuned'] = False
args_u['ms'] = ms
args_u['n_iters'] = n_iters 

##### create files for storing stim values
t_stim = ['storing/plastic_stim.csv']
initialize_file(t_stim[0], args_t)
u_stim = ['storing/constant_stim.csv']
initialize_file(u_stim[0], args_u)

## run networks in parallel
tt = PQ(args_t, stim_l = t_stim)
ut = PQ(args_u, stim_l = u_stim)

print('Time to run tuned ' + str(tt/60) + ' minutes')
print('Time to run untuned ' + str(ut/60) + ' minutes')
'''
############## PLOT STORED RESULTS #############

t_stim = ['storing/plastic_stim.csv'] ## comment out to plot new results
u_stim = ['storing/constant_stim.csv'] ## comment out to plot new results
 
## read in files
tuned_stims = pd.read_csv(t_stim[0])
tuned_stims = pd.DataFrame.transpose(tuned_stims)
constant_stims = pd.read_csv(u_stim[0])
constant_stims = pd.DataFrame.transpose(constant_stims)
ms = len(tuned_stims)

## create FEVER results (see Druckmann + Chlovskii 2014 for actual implementation details)
fever_stims = pd.DataFrame(np.ones((tuned_stims.shape[0], tuned_stims.shape[1])))

## plot Fig 2B
dfs = [tuned_stims, constant_stims,  fever_stims]
pa = plot_args(FEVER=True, rb=False, poster=True)
pa['labels'] = ['Plastic Random Synapse','Constant Random Synapse', 'FEVER']
pa['ylim'] = [-.1, 1.15]
pa['xspot'] = 380
pa['right_margins'] = False
pa['plt_errors'] = True
pa['yadds'] = [-.1, .04, .03]
pa['save'] = True
plot_pqs(dfs, pa)

## plot zoomed 2C
paz = plot_args(rb=False, poster=True)
dfs = [tuned_stims, fever_stims]
paz['labels'] = ['Plastic Random Synapse', 'FEVER']
paz['yadds'] = [.00022, .00003]
paz['ylim'] = [.9968, 1.0002]
paz['yadds'] = [.00055, -.0003]
paz['right_margins'] = False
paz['plt_errors'] = True
paz['xspot'] = 500
paz['save'] = True
paz['ylabel'] = ''
paz['colors'] = [pa['colors'][0], pa['colors'][2]]
plot_pqs(dfs, paz)


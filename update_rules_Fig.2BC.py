"""Recreates Figure 2BC from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NeuralNetwork import NN
from NetworkSimulation import Sim
from quantify_sims_multithreading import * 
import pandas as pd
import numpy as np
from arg_parsing import *
from plot_util import *

ms = 3000
n_iters = 100 
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
## add random str in to not over-write 
r = str(abs(np.round(np.random.randn(), 4)))  
t_stim = ['storing/plastic_stim_' + r + '.csv']
initialize_file(t_stim[0], args_t)
u_stim = ['storing/constant_stim_' + r + '.csv']
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
dfs = [constant_stims, tuned_stims, fever_stims]
pa = plot_args()
pa['colors'] = ['red', 'blue', 'black']
pa['labels'] = ['Constant Synapse', 'Plastic Synapse', 'FEVER']
pa['ylim'] = [-.1, 1.1]
pa['yadds'] = [0, -.02, .02]
pa['xspot'] = 1000
pa['right_margins'] = False
pa['yadds'] = [.05, -.1, .02]
plot_pqs(dfs, pa)


## plot zoomed 2C
dfs = [tuned_stims, fever_stims]
pa['labels'] = ['Plastic Synapse', 'FEVER']
pa['colors'] = ['blue', 'black']
pa['error_colors'] = ['#8080ff', '#8080ff']
pa['yadds'] = [.00018, .00005]
pa['ylim'] = [.9925, 1.003]
pa['yadds'] = [.0005, .0005]
pa['ylabel'] = ''
plot_pqs(dfs, pa)


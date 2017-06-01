"""Recreates Figure 3A from bioRxiv: 144683 (Federer & Zylberbeg 2017) """

from NeuralNetwork import NN
from NetworkSimulation import Sim
import pandas as pd
import numpy as np
import quantify_sims_multithreading as pq
from scipy import stats
from arg_parsing import get_args
import time
from the_generic_plotter import  plot_multidim 

def avg(x):
    """ Returns avg"""
    return sum(x) / len(x)

def initialize_file(loc, cols):
    """ initializes a file with column names """
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()

### command line arguments, all defaults set otherwise 
args = get_args()
if(args.n_stim <= 1): ### not multidim results if only one stim 
    args.n_stim = 4
r = str(np.random.randn())

### create files for storing stim values
stim_cols = list(range(args.ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])
t_stims = list()
u_stims = list()
for i in range(args.n_stim):
    l3 = 'storing/relu_stims_'+ str(i+1) + '_' + r + '.csv'
    t_stims.append(l3)
    initialize_file(l3, stim_cols)
    l4 = 'storing/constant_stims_'+ str(i+1) + '_' + r + '.csv'
    u_stims.append(l4)
    initialize_file(l4, stim_cols)
                
tt = pq.PQ(n_iters=args.n_iters, stim_l = t_stims, n_neurons=args.n_neurons,
    ms=args.ms, dt=args.dt, eta=args.eta, n_stim=args.n_stim, plastic_synapse=True)
ut = pq.PQ(n_iters=args.n_iters, stim_l = u_stims, n_neurons=args.n_neurons, ms=args.ms, dt=args.dt,n_stim=args.n_stim, plastic_synapse=False)

## run networks in parallel
print('Time to run tuned ' + str(tt/60) + ' minutes')
print('Time to run untuned ' + str(ut/60) + ' minutes')

## create lists for plots 
tdfs = list()
udfs = list()
tLabels = list()
uLabels=list()

for i in range(args.n_stim):
    ## open and read stim
    ts = pd.read_csv('storing/relu_stims_'+ str(i+1) + '_' + r + '.csv')
    tdfs.append(pd.DataFrame.transpose(ts))
    us = pd.read_csv('storing/constant_stims_'+ str(i+1) + '_' + r + '.csv')
    udfs.append(pd.DataFrame.transpose(us))
    tLabels.append('Plastic Random Synapse S' + str(i+1))
    uLabels.append('Constant Random Synapse S' + str(i+1))

## plot Fig 3A  
colors = ['orangered', 'blue', 'green', 'purple']
a_colors = ['#ff9999','#8080ff', '#c6ecd7','#ccb3e6']
plot_multidim(tdfs, udfs, tLabels, uLabels, colors=colors,
             a_colors = a_colors, yadds=[-.1, 0, -.05, .05], ylim=[-.15, 1.1])



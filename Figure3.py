from NeuralNetwork import NN
from NetworkSimulation import Sim
import pandas as pd
import numpy as np
import quantify_sims_multithreading as pq
from scipy import stats

import time 

def avg(x):
    return sum(x) / len(x)

def initialize_file(loc, cols):
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()

## network params 
n_stim = 4
n_neurons = 100
ms = 100
n_iters = 2
dt = .001
eta = .0001
r = str(np.random.randn())

stim_cols = list(range(ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])
t_stims = list()
u_stims = list()
for i in range(n_stim):
    l3 = 'storing/relu_stims_'+ str(i+1) + '_' + r + '.csv'
    t_stims.append(l3)
    initialize_file(l3, stim_cols)
    l4 = 'storing/constant_stims_'+ str(i+1) + '_' + r + '.csv'
    u_stims.append(l4)
    initialize_file(l4, stim_cols)
                
tt = pq.PQ(n_iters=n_iters, stim_l = t_stims, n_neurons=n_neurons,
    ms=ms, dt=dt, eta=eta, n_stim=4, plastic_synapse=True)
ut = pq.PQ(n_iters=n_iters, stim_l = u_stims, n_neurons=n_neurons, ms=ms, dt=dt,n_stim=4, plastic_synapse=False)

print('Time to run tuned ' + str(tt/60) + ' minutes')
print('Time to run untuned ' + str(ut/60) + ' minutes')

tdfs = list()
udfs = list()
tLabels = list()
uLabels=list()
dfs = list()
labels = list()

for i in range(n_stim):
    ## open and read stim
    ts = pd.read_csv('storing/relu_stims_'+ str(i+1) + '_' + r + '.csv')
    tdfs.append(pd.DataFrame.transpose(ts))
    dfs.append(pd.DataFrame.transpose(ts))
    us = pd.read_csv('storing/constant_stims_'+ str(i+1) + '_' + r + '.csv')
    udfs.append(pd.DataFrame.transpose(us))
    dfs.append(pd.DataFrame.transpose(us))
    ## append to phi list
    tLabels.append('Plastic Random Synapse S' + str(i+1))
    labels.append('Plastic Random Synapse S' + str(i+1))
    uLabels.append('Constant Random Synapse S' + str(i+1))
    labels.append('Constant Random Synapse S' + str(i+1))
  
colors = ['orangered', 'blue', 'green', 'purple']
a_colors = ['#ff9999','#8080ff', '#c6ecd7','#ccb3e6']

import sys
sys.path.insert(0, 'plotting/')
from the_generic_plotter import  plot_multidim 

plot_multidim(tdfs, udfs, tLabels, uLabels, colors=colors, xspot = 3000, 
             a_colors = a_colors, yadds=[-.1, 0, -.05, .05], ylim=[-.15, 1.1])



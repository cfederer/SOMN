"""Recreates Figure 6 from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NeuralNetwork import NN
from NetworkSimulation import Sim
import pandas as pd
import numpy as np
import quantify_sims_multithreading as pq
from arg_parsing import get_args
from the_generic_plotter import plot_stims 

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

##other params 
connectivities = [.1, .2, .3, .4, .5, .6, .7, .8 ,.9]
total_time = 0

## create files for storing stim 
r = str(np.random.randn())
stim_cols = list(range(args.ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])


## run networks in parallel 
for i in range(len(connectivities)):
    connectivity = connectivities[i]
    print('Running network ' + str(connectivity*100) + '% connected')
    f_stim = ['storing/relu_stim_connectivity' + str(connectivity) + '_' +  r + '.csv']
    initialize_file(f_stim[0], stim_cols)
    t = pq.PQ(n_iters = args.n_iters, stim_l = f_stim, n_neurons = args.n_neurons, ms=args.ms,
              plastic_synapse=True, dt=args.dt, eta=args.eta, connectivity=connectivity)
    print('Time to run ' + str(connectivity*100) + '% connected network ' + str(t/60) + ' minutes')
    total_time += t
    
print('Total time to run ' + str(total_time/60) + ' minutes')


### read in files 
stims_list = list()
labels = list()
xlabels = list()

for i in range(len(connectivities)):
    connectivity = connectivities[i]
    f_stim = 'storing/relu_stim_connectivity' + str(connectivity) + '_' +  r + '.csv'
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    stims_list.append(stims)
    labels.append(str(int(connectivity*100))+ '% Connected')
    xlabels.append(str(int(connectivity*100)) + '%')

### add fully connected network info
f_stim = 'storing/t_stim_0.639587767357.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
stims_list.append(stims[:args.ms])
labels.append('100% Connected')
xlabels.append('100%')


plot_stims(stims_list, labels=labels,  ylim=[.94, 1.002], 
                #   10    20      30      40      50     60      70      80     90      100
           yadds=[-.001,-.0015, -.0023, -.0044, -.005, -.0049, -.0031,    0,    .0015,  .003])




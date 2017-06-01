"""Recreates Figure 2BC from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NeuralNetwork import NN
from NetworkSimulation import Sim
import quantify_sims_multithreading as pq
from the_generic_plotter import plot_stims, plot_activities
import pandas as pd
import numpy as np
from arg_parsing import get_args

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

### create files for storing stim values 
stim_cols = list(range(args.ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])
r = str(np.random.randn())
t_stim = ['storing/relu_stim_' + r + '.csv']
initialize_file(t_stim[0], stim_cols)
u_stim = ['storing/constant_stim_' + r + '.csv']
initialize_file(u_stim[0], stim_cols)

## run networks in parallel
tt = pq.PQ(n_iters=args.n_iters, stim_l = t_stim, 
           n_neurons=args.n_neurons, ms=args.ms, dt=args.dt, eta=args.eta, plastic_synapse=True, pretune=False)
ut = pq.PQ(n_iters=args.n_iters, stim_l = u_stim,
           n_neurons=args.n_neurons, ms=args.ms, dt=args.dt, plastic_synapse=False)

print('Time to run tuned ' + str(tt/60) + ' minutes')
print('Time to run untuned ' + str(ut/60) + ' minutes')

## read in files
tuned_stims = pd.read_csv(t_stim[0])
tuned_stims = pd.DataFrame.transpose(tuned_stims)
constant_stims = pd.read_csv(u_stim[0])
constant_stims = pd.DataFrame.transpose(constant_stims)

## create FEVER results (see Druckmann + Chlovskii 2014 for actual implementation details)
fever_stims = pd.DataFrame(np.ones((tuned_stims.shape[0], tuned_stims.shape[1])))

## plot 2B
labels = ['Constant Synapse', 'Plastic Synapse', 'FEVER']
dfs = [constant_stims, tuned_stims, fever_stims]
plot_stims(dfs, labels=labels, colors=['red', 'blue', 'black'], yadds=[.025, -.08, .025], ylim=[-.1, 1.1])

## plot zoomed 2C
dfs = [tuned_stims, fever_stims]
labels = ['Plastic Synapse', 'FEVER']
plot_stims(dfs, labels=labels, colors=['blue', 'black'], a_colors = ['#8080ff', '#8080ff'], yadds=[.00018, .00005])

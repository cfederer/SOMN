from NeuralNetwork import NN
from NetworkSimulation import Sim
import quantify_sims_multithreading as pq
from the_generic_plotter import plot_stims, plot_activities
import pandas as pd
import numpy as np 

def avg(x):
    return sum(x) / len(x)

def initialize_file(loc, cols):
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()

## set network parameters 
n_neurons = 100
ms = 3000
n_iters = 100
dt = .001
n_stim=1
eta = .0001
stim_cols = list(range(ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])
r = str(np.random.randn())

## initialize files for storing 
t_stim = ['storing/relu_stim_' + r + '.csv']
initialize_file(t_stim[0], stim_cols)
u_stim = ['storing/constant_stim_' + r + '.csv']
initialize_file(u_stim[0], stim_cols)

## run networks in parallel
tt = pq.PQ(n_iters=n_iters, stim_l = t_stim, 
           n_neurons=n_neurons, ms=ms, dt=dt, eta=eta, plastic_synapse=True, pretune=False)
ut = pq.PQ(n_iters=n_iters, stim_l = u_stim,
           n_neurons=n_neurons, ms=ms, dt=dt, plastic_synapse=False)

print('Time to run tuned ' + str(tt/60) + ' minutes')
print('Time to run untuned ' + str(ut/60) + ' minutes')

## read in files
tuned_stims = pd.read_csv(t_stim[0])
tuned_stims = pd.DataFrame.transpose(tuned_stims)
constant_stims = pd.read_csv(u_stim[0])
constant_stims = pd.DataFrame.transpose(constant_stims)

## create FEVER results (see Druckmann + Chlovskii 2014 for actual implementation details)
fever_stims = pd.DataFrame(np.ones((tuned_stims.shape[0], tuned_stims.shape[1])))

## plot 
labels = ['Constant Synapse', 'Plastic Synapse', 'FEVER']
dfs = [constant_stims, tuned_stims, fever_stims]
plot_stims(dfs, labels=labels, colors=['red', 'blue', 'black'], yadds=[.025, -.08, .025], ylim=[-.1, 1.1])

## plot zoomed 
dfs = [tuned_stims, fever_stims]
labels = ['Plastic Synapse', 'FEVER']
plot_stims(dfs, labels=labels, colors=['blue', 'black'], a_colors = ['#8080ff', '#8080ff'], yadds=[.00018, .00005])

from NeuralNetwork import NN
from NetworkSimulation import Sim
import pandas as pd
import numpy as np
import quantify_sims_multithreading as pq
from scipy import stats

def avg(x):
    return sum(x) / len(x)

def initialize_file(loc, cols):
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()
    
n_neurons = 100
ms = 10
n_iters = 2
dt = .001
eta = .0001
n_stim = 1
stim_cols = list(range(ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])
r = str(np.random.randn())

rw_stim = ['storing/rw_stim_' + r + '.csv']
initialize_file(rw_stim[0], stim_cols)

t_rw = pq.PQ(n_iters = n_iters, stim_l = rw_stim, n_neurons = n_neurons,
          ms=ms, dt=dt, eta=eta, plastic_synapse=True, rws=True)
print('Time to run ' + str(t_rw/60) + ' minutes')

t_stim = ['storing/t_stim_0.639587767357.csv']
u_stim = ['storing/u_stim_0.265785068576.csv']
rw_stim = ['storing/rw_stim_' + r + '.csv']
tuned_stims = pd.read_csv(t_stim[0])
tuned_stims = pd.DataFrame.transpose(tuned_stims)
rw_stims = pd.read_csv(rw_stim[0])
rw_stims = pd.DataFrame.transpose(rw_stims)
constant_stims = pd.read_csv(u_stim[0])
constant_stims = pd.DataFrame.transpose(constant_stims)
dfs = [constant_stims[:ms],rw_stims , tuned_stims[:ms]]
labels = ['Constant Random Synapse', 'Plastic Synapse (Random Feedback)', 'Plastic Synapse (Symmetric Feedback)']
colors = ['red', 'green', 'blue']
a_colors = ['#ff9999', '#4dff4d' , '#8080ff']

from the_generic_plotter import plot_stims

plot_stims(dfs, labels = labels, colors=colors, a_colors=a_colors, xspot = 100,ylim=[-.1, 1.01],
           yadds=[-.07, -.07, -.07])

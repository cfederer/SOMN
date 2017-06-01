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

connectivities = [.1, .2, .3, .4, .5, .6, .7, .8 ,.9]

n_neurons = 100
ms = 10
iters = 2
dt = .001
eta = .0001
n_stim = 1 
total_time = 0
r = str(np.random.randn())
stim_cols = list(range(ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])

for i in range(len(connectivities)):
    connectivity = connectivities[i]
    print('Running network ' + str(connectivity*100) + '% connected')
    f_stim = ['storing/relu_stim_connectivity' + str(connectivity) + '_' +  r + '.csv']
    initialize_file(f_stim[0], stim_cols)
    t = pq.PQ(n_iters = iters, stim_l = f_stim, n_neurons = n_neurons, ms=ms,
              plastic_synapse=True, dt=dt, eta=eta, connectivity=connectivity)
    print('Time to run ' + str(connectivity*100) + '% connected network ' + str(t/60) + ' minutes')
    total_time += t
    
print('Total time to run ' + str(total_time/60) + ' minutes')

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
stims_list.append(stims[:ms])
labels.append('100% Connected')
xlabels.append('100%')

from the_generic_plotter import plot_stims 
plot_stims(stims_list, labels=labels, xspot = 3000, ylim=[.94, 1.002], 
                #   10    20      30      40      50     60      70      80     90      100
           yadds=[-.001,-.0015, -.0023, -.0044, -.005, -.0049, -.0031,    0,    .0015,  .003])




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

fracs_tuned = [.1, .2, .3, .4 , .5, .6, .7, .8 ,.9]
total_time = 0 
activation = 'relu'
n_neurons = 100
ms = 3000
iters = 2
dt = .001
eta = .0001
n_stim = 1
r = str(np.random.randn())
stim_cols = list(range(ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])

for i in range(len(fracs_tuned)):
    frac = fracs_tuned[i]
    print('Running ' + str(frac) + ' frac tuned ')
    f_stim = ['storing/relu_stim_fracL_' + str(frac) + '_' +  r + '.csv']
    initialize_file(f_stim[0], stim_cols)
    t = pq.PQ(n_iters=iters, stim_l = f_stim, n_neurons=n_neurons,
              ms=ms, dt=dt, eta=eta, plastic_synapse=True, activation=activation, frac_tuned = frac)
    print('Time to run frac:' + str(fracs_tuned[i]) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')

### plot stored data

stims_list = list()
labels = list()
xlabels = list()

for i in range(len(fracs_tuned)):
    frac = fracs_tuned[i]
    f_stim = 'storing/relu_stim_fracL_' + str(frac) + '_' +  r + '.csv'
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    stims_list.append(stims)
    labels.append(str(int(fracs_tuned[i]*100)) + '% Plastic')
    xlabels.append(str(int(fracs_tuned[i]*100)) + '%')

## add fully tuned network
f_stim = 'storing/t_stim_0.639587767357.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
stims_list.append(stims[:ms])
labels.append('100% Plastic')
xlabels.append('100%')


from the_generic_plotter import plot_stims
save=True
fracs_tuned = [.1, .2, .3, .4 , .5, .6, .7, .8 ,.9, 1]

plot_stims(stims_list, labels=labels,  ylim = [.96, 1.0])

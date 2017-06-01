from NeuralNetwork import NN
from NetworkSimulation import Sim
import sys
import pandas as pd
import numpy as np
import quantify_sims_multithreading as pq
from scipy import stats
import time 
colors = ['#8B0000', '#FF0000','#FF4500', '#FF8C00', '#FFD700', '#7CFC00',  '#3CB371', '#20B2AA', '#00BFFF' , '#4169E1','#0000FF']

def avg(x):
    return sum(x) / len(x)

def initialize_file(loc, cols):
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()

fracs_noise= [.1, .2, .3, .4 , .5, .6, .7, .8 ,.9, 1]   
total_time = 0 
activation = 'relu'
n_neurons = 100
ms = 10
iters = 2
dt = .001
eta = .0001
n_stim = 1
r = str(np.random.randn())
stim_cols = list(range(ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])

for i in range(len(fracs_noise)):
    frac = fracs_noise[i]
    print('Running ' + str(frac) + ' frac noise ')
    f_stim = ['storing/relunoise_stim_' + str(frac) + '_' +  r + '.csv']
    initialize_file(f_stim[0], stim_cols)
    t = pq.PQ(n_iters=iters,stim_l = f_stim, n_neurons=n_neurons, ms=ms, plastic_synapse=True,
                         L_noise = frac, dt=dt, eta=eta)
    print('Time to run frac:' + str(frac) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')

stims_list = list()
labels = list()
## add 0 noise
f_stim = 'storing/t_stim_0.639587767357.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
stims_list.append(stims[:ms])
labels.append('α = 0')

for i in range(len(fracs_noise)):
    frac = fracs_noise[i]
    f_stim = 'storing/relunoise_stim_' + str(frac) + '_' +  r + '.csv'
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    stims_list.append(stims)
    labels.append('α = '+str(fracs_noise[i]))
    
fracs_noise= [0, .1, .2, .3, .4 , .5, .6, .7, .8 ,.9, 1]
d_colors = ['DarkRed', 'red','orangered', 'darkorange',
            'limegreen', 'teal', 'DodgerBlue',
            'blue','darkblue', 'mediumpurple', 'purple']

da_colors = ['#ff3333', '#ff8080','#ffb499', '#ffd199',
             '#c1f0c1', '#ccffff', '#b3d9ff', 
             '#8080ff', '#6666ff', '#c2adeb', '#ccb3e6']

from the_generic_plotter import plot_stims
save = True
plot_stims(stims_list, labels=labels,
           yadds=[.00008, -.0001, .0002, -.00023, 0, .00017, -.00006, 0, 0, .0001, -.00008], 
           xspot = 3000, colors=d_colors, a_colors = da_colors)

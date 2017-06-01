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
pretunes = [1, 5, 10]
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

for i in range(len(pretunes)):
    pretune = pretunes[i]
    print('Running ' + str(pretune) + ' training sessions ')
    f_stim = ['storing/relupretune_turnoff_stim_' + str(pretune) + '_' +  r + '.csv']
    initialize_file(f_stim[0], stim_cols)
    t = pq.PQ(n_iters=iters,stim_l = f_stim, n_neurons=n_neurons, ms=ms,
              plastic_synapse=False, dt=dt, eta=eta, pretune=pretune)
    print('Time to run frac:' + str(pretune) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')

stims_list = list()
labels = list()
## add no previous tunings (constant model)
f_stim = 'storing/u_stim_0.265785068576.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
stims_list.append(stims[:ms])
labels.append('Constant Pre-Trained Synapse (0)')

for i in range(len(pretunes)):
    pretune = pretunes[i]
    f_stim = 'storing/relupretune_turnoff_stim_' + str(pretune) + '_' +  r + '.csv'
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    stims_list.append(stims)
    labels.append('Constant Pre-Trained Synapse (' + str(pretune) + ')')

colors = ['orangered', 'blue', 'green', 'purple']
a_colors = ['#ff9999','#8080ff', '#c6ecd7','#ccb3e6']
import sys
sys.path.insert(0, 'plotting/')
from the_generic_plotter import plot_stims
plot_stims(stims_list, labels, colors=colors,a_colors=a_colors, yadds = [0, -.04, 0, 0], xspot=3000, ylim=[-.1, 1])


from NeuralNetwork import NN
from NetworkSimulation import Sim
import pandas as pd
import numpy as np
import parallel_quantify_nostoring as pq
from scipy import stats

def avg(x):
    return sum(x) / len(x)

def initialize_file(loc, cols):
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()

'''    
n_neurons = 100
ms = 100
n_iters = 10
dt = .001
eta = .0001
stim_cols = list(range(ms))
stim_cols = ','.join(['%.5f' % num for num in stim_cols])
r = str(np.random.randn())

t_phis = ['storing/relu_phis_' + r + '.csv']
initialize_phifile(t_phis[0])
u_phis = ['storing/constant_phis_' + r + '.csv']
initialize_phifile(u_phis[0])
t_stim = ['storing/relu_stim_' + r + '.csv']
initialize_stimfile(t_stim[0], stim_cols)
u_stim = ['storing/constant_stim_' + r + '.csv']
initialize_stimfile(u_stim[0], stim_cols)

tt = pq.PQ(n_iters=n_iters, stim_l = t_stim, phi_l = t_phis, 
           n_neurons=n_neurons, ms=ms, dt=dt, eta=eta, plastic_synapse=True, pretune=False)
ut = pq.PQ(n_iters=n_iters, stim_l = u_stim, phi_l = u_phis,
           n_neurons=n_neurons, ms=ms, dt=dt, plastic_synapse=False)

print('Time to run tuned ' + str(tt/60) + ' minutes')
print('Time to run untuned ' + str(ut/60) + ' minutes')
#ft = pq.PQ(n_iters=n_iters, n_neurons=n_neurons, ms=ms, dt=dt, plastic_synapse=False, FEVER=True)
'''

t_phis = ['storing/t_phis_0.639587767357.csv']
u_phis = ['storing/u_phis_0.265785068576.csv']
t_stim = ['storing/t_stim_0.639587767357.csv']
u_stim = ['storing/u_stim_0.265785068576.csv']

save=True
tuned_stims = pd.read_csv(t_stim[0])
tuned_stims = pd.DataFrame.transpose(tuned_stims)
constant_stims = pd.read_csv(u_stim[0])
constant_stims = pd.DataFrame.transpose(constant_stims)
tuned_phis = pd.read_csv(t_phis[0])
constant_phis = pd.read_csv(u_phis[0])

fever_stims = pd.DataFrame(np.ones((tuned_stims.shape[0], tuned_stims.shape[1])))
fever_phis = pd.DataFrame(np.repeat(0, len(tuned_phis)))

import sys
sys.path.insert(0, 'plotting/')
from plots import plot_table
from the_generic_plotter import plot_stims 
labels = ['Constant Synapse', 'Plastic Synapse', 'FEVER']
dfs = [constant_stims, tuned_stims, fever_stims]
avg_phis = [avg(constant_phis['0']), avg(tuned_phis['0']), avg(fever_phis[0])]
errs = [stats.sem(constant_phis['0']), stats.sem(tuned_phis['0']), stats.sem(fever_phis[0])]

plot_stims(dfs, folder='Fig2', labels=labels,show=True, save=save,
          key='reluvfevervuntuned', colors=['red', 'blue', 'black'], yadds=[.025, -.08, .025], ylim=[-.1, 1.1])

dfs = [tuned_stims, fever_stims]
avg_phis = [avg(tuned_phis['0']), avg(fever_phis[0])]
errs = [stats.sem(tuned_phis['0']), stats.sem(fever_phis[0])]
labels = ['Plastic Synapse', 'FEVER']
plot_stims(dfs, folder='Fig2', labels=labels,show=True, save=save,
           key='reluvfevervuntuned', colors=['blue', 'black'], a_colors = ['#8080ff', '#8080ff'], yadds=[.00018, .00005])
#plot_table(avg_phis, errors=errs, colLabels = labels,show=True,folder='Fig2', save=save)


"""Not same data as in arxiv paper but same methods as Figure 5A """
from NeuralNetwork import NN
from NetworkSimulation import Sim
from quantify_sims_multithreading import * 
import pandas as pd
import numpy as np
from arg_parsing import *
from plot_util import *


ms = 3000
n_iters = 100
pretunes = [1, 5, 10]
############## UNCOMMENT TO RUN NEW RESULTS #############
'''
args = get_args()
args['tuned'] = True 
args['ms'] = ms
args['n_iters'] = n_iters
total_time = 0
r = str(abs(np.round(np.random.randn(), 4))) 
## run networks in parallel
for i, n in enumerate(pretunes):
    args['pretune'] = n
    print('Running ' + str(n) + ' training sessions ')
    f_stim = ['storing/pretrain_plastic_ntrains:' + str(n) + '_' + r + '.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run frac:' + str(n) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')
'''


dfs = list()
labels = list()

## add no previous tunings 
f_stim = 'storing/plastic_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
dfs.append(stims[:ms])
labels.append('Constant Pre-Trained Synapse (0)')

### read in files 
for i, n in enumerate(pretunes):
    f_stim = 'storing/pretrain_plastic_ntrains:' + str(n) + '.csv' ## comment out to plot new results 
    #f_stim = 'storing/pretrain_plastic_ntrains:' + str(n) + '_' + r + '.csv'  ##uncomment to plot new results
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append('Plastic Pre-Trained Synapse (' + str(n) + ')')
pa = plot_args()
pa['colors'] = ['orangered', 'blue', 'green', 'purple']
pa['error_colors'] = ['#ff9999','#8080ff', '#c6ecd7','#ccb3e6']
pa['labels'] = labels 
plot_pqs(dfs, pa)

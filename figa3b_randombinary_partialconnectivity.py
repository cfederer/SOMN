from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *

ms = 3000
n_iters = 10
connectivities = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
############################# UNCOMMENT TO RUN NEW RESULTS #############################
'''
args = get_args()
args['tuned'] = True
args['ms'] = ms
args['n_iters'] = n_iters
args['error'] = 'binary'
args['rws'] = True

total_time = 0 

for i, c in enumerate(connectivities):
    args['connectivity'] = c
    print('Running ' + str(c) + ' connectivity ')
    f_stim = ['storing/binary_partial_connectivity_' + str(c) +'.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run frac:' + str(c) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')
'''
############################# PLOTTING STORED RESULTS #############################

dfs = list()
labels = list()
for i, c in enumerate(connectivities):
    f_stim = ['storing/binary_partial_connectivity_' + str(c) +'.csv']
    stims = pd.read_csv(f_stim[0])
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(' ' + str(int(c*100)) + '% ')

## add 100% Plastic data   
f_stim = 'storing/binary_random.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
dfs.append(stims[:ms])
labels.append(' 100% ')

pa = plot_args(n_curves=len(dfs))
ordered_labels = [' 10%', ' 20%', ' 40%', ' 30%',' 70%', ' 50%', ' 90%', ' 100%', ' 60%', ' 80%']
pa['labels'] = ordered_labels 
pa['plt_errors'] = True
pa['xspot'] = 3000
pa['right_margins'] = True
pa['xspot'] = 3000
pa['ylim'] = [0, 1.2]
t = 1.1
g = .08
pa['title'] = 'Partial Connectivity'
pa['title_size'] = pa['label_size']
pa['yspots'] = [t-9*g, t-8*g, t-7*g, t-6*g, t-5*g, t-4*g, t-3*g, t-2*g, t-g, t]
plot_pqs(dfs, pa)

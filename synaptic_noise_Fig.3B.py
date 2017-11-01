"""Recreates Figure 3A from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NeuralNetwork import NN
from NetworkSimulation import Sim
from quantify_sims_multithreading import * 
import pandas as pd
import numpy as np
from arg_parsing import *
from plot_util import *


ms = 3000
alphas = [.2, .4, .6, .8 , 1]
n_iters = 100
############################# UNCOMMENT TO RUN NEW RESULTS #############################
'''
args = get_args()
args['ms'] = ms
args['n_iters'] = n_iters
args['tuned'] = True 

r = str(abs(np.round(np.random.randn(), 4))) 
total_time = 0

for i, a in enumerate(alphas):
    args['L_noise'] = a
    print('Running ' + str(a) + ' frac noise ')
    f_stim = ['storing/synaptic_noise_alpha=' + str(a) +'_' + r + '.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run frac:' + str(a) + ' ' + str(t/60) + ' minutes')
    total_time+=t
                
print('Total time to run ' + str(total_time/60) + ' minutes')
'''
dfs = list()
labels = list()

## add no noise
f_stim = 'storing/plastic_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
labels.append('α = 0')
dfs.append(stims[:ms])


## read in files 
for i, a in enumerate(alphas):
    f_stim = ['storing/synaptic_noise_alpha=' + str(a) +'.csv'] ## comment out to plot new results
    #f_stim = ['storing/synaptic_noise_alpha=' + str(a) +'_' + r + '.csv'] ## uncomment to plot new results
    stims = pd.read_csv(f_stim[0])
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append('α = '+str(a))

### plot
pa = plot_args()
pa['labels'] = labels
pa['xspot'] = 2000
pa['colors'] = ['red', 'darkorange',
            'limegreen', 'teal', 'DodgerBlue',
            'blue','darkblue', 'purple', 'black']
pa['ylim'] = [.9945, 1.000]
pa['yadds'] = [-.0004, .0001, .0001, .00003, .0001, .00006]
plot_pqs(dfs, pa)

"""Recreates Figure 4A from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NeuralNetwork import NN
from NetworkSimulation import Sim
from quantify_sims_multithreading import * 
import pandas as pd
import numpy as np
from arg_parsing import *
from plot_util import *


ms = 3000
n_iters = 100
args = get_args()
args['tuned'] = True
args['ms'] = ms
args['n_iters'] = n_iters
fracs_tuned = [.001, .01, .1, .2, .3, .5, .7, .9]


############## UNCOMMENT TO RUN NEW RESULTS #############
'''
total_time = 0 
### create files for storing stim values 
r = str(abs(np.round(np.random.randn(), 4)))

for i, f in enumerate(fracs_tuned):
    args['connectivity'] = f
    print('Running ' + str(f) + ' frac tuned ')
    f_stim = ['storing/partial_plasticity_' + str(f) + '_' +  r + '.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run frac:' + str(f) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')
'''


######################## PLOT FULL ########################
dfs = list()
fracs_tuned = [.001, .01, .1, .2, .3, .5, .7, .9]
for i, f in enumerate(fracs_tuned):
    f_stim = 'storing/partial_plasticity_' + str(f) + '.csv' ## comment out to plot new results
    #f_stim = 'storing/partial_plasticity_' + str(f) + '_' +  r + '.csv' ##uncomment to plot new results 
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)

## add 100% Plastic data   
f_stim = 'storing/plastic_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
dfs.append(stims[:ms])

pa = plot_args()
pa['labels'] = ['.1% Plastic', '1% Plastic', '10% Plastic', '', '', '', '', '', '100% Plastic']
pa['yadds'] = [.02, .03, -.07, 0, 0, 0, 0, 0, .02]
pa['plt_errors'] = True
pa['text_size'] = 15
pa['xspot'] = 1500
pa['ylim'] = [-.1, 1.1]
plot_pqs(dfs, pa)

######################## PLOT Zoomed ########################
dfs = list()
labels = list()
fracs_tuned = [.1, .2, .3, .5, .7, .9]
for i, f in enumerate(fracs_tuned):
    f_stim = 'storing/partial_plasticity_' + str(f) + '.csv'
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(str(int(f*100)) + '% Plastic')

pa['labels'] = labels 
pa['yadds'] = [-.001, -.0015, -.0015, -.0015, -.0025, .0015]
pa['plt_errors'] = True
pa['text_size'] = 15
pa['xspot'] = 3000
pa['ylim'] = [.91, 1.01]
pa['colors'] = ['orangered', 'darkorange',
            'limegreen', 'teal', 'DodgerBlue',
            'blue','darkblue', 'purple', 'black']
pa['error_colors'] = ['#ffb499', '#ffd199',
             '#c1f0c1', '#ccffff', '#b3d9ff', 
             '#8080ff', '#6666ff','#ccb3e6', 'gray']
pa['ylabel'] = ''
pa['right_margins'] = True 
plot_pqs(dfs, pa)

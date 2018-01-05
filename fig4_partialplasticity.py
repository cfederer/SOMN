"""Recreates Figure 4A from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *


ms = 3000
n_iters = 10
args = get_args()
args['tuned'] = True
args['ms'] = ms
args['n_iters'] = n_iters
fracs_tuned = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9]


############## UNCOMMENT TO RUN NEW RESULTS #############
'''
total_time = 0 
### create files for storing stim values 
for i, f in enumerate(fracs_tuned):
    args['frac_tuned'] = f
    print('Running ' + str(f) + ' frac tuned ')
    f_stim = ['storing/partial_plasticity_' + str(f) +'.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run frac:' + str(f) + ' ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')
'''

######################## PLOT FULL ########################

dfs = list()
labels = list()
fracs_tuned = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9]
for i, f in enumerate(fracs_tuned):
    f_stim = 'storing/partial_plasticity_' + str(f) + '.csv' ## comment out to plot new results
    #f_stim = 'storing/partial_plasticity_' + str(f) + '_' +  r + '.csv' ##uncomment to plot new results 
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(' ' + str(int(f*100)) + '%')

## add 100% Plastic data   
f_stim = 'storing/plastic_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
dfs.append(stims[:ms])
labels.append('100% Plastic')

pa = plot_args(n_curves=len(dfs))
pa['labels'] = ['.1% Plastic', '1% Plastic', '10% Plastic', '', '', '', '', '', '', '', '', '100% Plastic']
pa['yadds'] = [.05, -.15, -.1, 0, 0, 0, 0, 0, 0, 0, 0, .03]
pa['plt_errors'] = True
pa['xspot'] = 1500
pa['ylim'] = [-.1, 1.16]
plot_pqs(dfs, pa)

######################## PLOT Zoomed ########################

labels = list()
for i, f in enumerate(fracs_tuned):
    labels.append(' ' + str(int(f*100)) + '%')

pa['labels'] = ordered_labels(dfs[2:11], labels[2:11])
t = .999
g = .003
pa['yspots'] = [.965, t-7*g, t-6*g, t-5*g, t-4*g, t-3*g, t-2*g, t-g, t]
pa['plt_errors'] = True
pa['xspot'] = 3000
pa['ylim'] = [.96, 1.001]
pa['colors'] = pa['colors'][2:11]
pa['ylabel'] = ''
pa['right_margins'] = True
plot_pqs(dfs[2:11], pa)


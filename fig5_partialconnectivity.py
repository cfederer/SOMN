from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *

ms = 3000
n_iters = 10
args = get_args()
args['ms'] = ms
args['n_iters'] = n_iters
args['tuned'] = True
connectivities = [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9]

############## UNCOMMENT TO RUN NEW RESULTS #############
'''
total_time = 0
## run networks in parallel 
for i, c in enumerate(connectivities):
    print('Running network ' + str(c*100) + '% connected')
    f_stim = ['storing/connectivity_' + str(c) + '.csv']
    args['connectivity'] = c 
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run ' + str(c*100) + '% connected network ' + str(t/60) + ' minutes')
    total_time += t
    
print('Total time to run ' + str(total_time/60) + ' minutes')
'''
### read in files
dfs = list()
labels = list()

for i, c in enumerate(connectivities):
    f_stim = ['storing/connectivity_' + str(c)+'.csv'] 
    stims = pd.read_csv(f_stim[0])
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(' ' + str(int(100*c)) + '%')

## add fully tuned network
f_stim = 'storing/plastic_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
dfs.append(stims[:ms])
labels.append(' 100%')


#### plot full
pa = plot_args(n_curves = len(dfs))
pa['labels'] = ['1% Connected', '10% Connected','', '', '', '', '', '', '', '', '100% Connected']
pa['yadds'] = [.02, -.09, 0 , 0 , 0, 0, 0, 0, 0, 0, .02]
pa['show'] = True
pa['plt_errors'] = True
pa['xspot'] = 1000
pa['ylim'] = [-.1, 1.15]
plot_pqs(dfs, pa)

#### plot zoomed
paz = plot_args(n_curves = len(dfs[1:10]))
paz['labels'] = ordered_labels(dfs[1:10], labels[1:10])
paz['ylim'] = [.94, 1.001]
paz['colors'] = pa['colors'][1:10]
t = .997
g = .004
paz['yspots'] = [.947, t-7*g, t-6*g, t-5*g, t-4*g, t-3*g, t-2*g, t-g, t]
paz['xspot'] = 3000
paz['plt_errors']=True
paz['right_margins'] = True
paz['show'] = True
paz['ylabel'] = '' 
plot_pqs(dfs[1:10], paz)

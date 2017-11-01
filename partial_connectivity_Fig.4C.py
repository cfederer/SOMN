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
args['ms'] = ms
args['n_iters'] = n_iters
args['tuned'] = True
connectivities = [.1, .3, .5, .7 ,.9]
############## UNCOMMENT TO RUN NEW RESULTS #############
'''

total_time = 0

## create files for storing stim 
r = str(abs(np.round(np.random.randn(), 4)))

## run networks in parallel 
for i, c in enumerate(connectivities):
    print('Running network ' + str(c*100) + '% connected')
    f_stim = ['storing/connectivity_' + str(c) + '_' +  r + '.csv']
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
    f_stim = ['storing/connectivity_' + str(c)+'.csv'] ## comment out to plot new results
    #f_stim = ['storing/connectivity_' + str(c) + '_' +  r + '.csv'] ## uncomment to plot new results 
    stims = pd.read_csv(f_stim[0])
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(str(int(100*c)) + '% Connected')

## add fully tuned network
f_stim = 'storing/plastic_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
dfs.append(stims[:ms])
labels.append('100% Connected')

pa = plot_args()
pa['labels'] = labels
pa['show'] = True
pa['right_margins'] = True
pa['ylim'] = [.88, 1.01]
pa['plt_errors'] = True
pa['colors'] = ['DarkRed', 'darkorange',
            'limegreen', 'teal', 'DodgerBlue',
            'blue','darkblue', 'purple', 'black']
pa['error_colors'] = ['#ff3333', '#ffd199',
             '#c1f0c1', '#ccffff', '#b3d9ff', 
             '#8080ff', '#6666ff','#ccb3e6', 'gray']
pa['yadds'] = [-.002, -.006, -.0128, -.0079, -.002, .006]
pa['xspot'] = 3000
pa['text_size'] = 18
pa['label_size'] = 20
plot_pqs(dfs, pa)




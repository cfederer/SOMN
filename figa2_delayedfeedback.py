from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *

ms = 3000
n_iters = 10
delays = [10, 20, 40, 50]
pretune = 5 
############## UNCOMMENT TO RUN NEW RESULTS #############
'''
args = get_args()
args['tuned'] = True 
args['ms'] = ms
args['n_iters'] = n_iters
args['delay'] = True
args['pretune'] = pretune 
args['eta'] = args['eta'] / 100


## run networks in parallel
total_time = 0
for d in delays:
    args['delay_ms'] = d 
    print('Running ' + str(d) + ' ms delay ')
    f_stim = ['storing/delayed_pretrain:' + str(pretune) + '_delay' + str(d) +'.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run ' + str(d) + ' ms delay: ' + str(t/60) + ' minutes')
    total_time+=t

print('Total time to run ' + str(total_time/60) + ' minutes')
'''
### plot
dfs = list()
labels = list()
### read in files
for d in delays:
    f_stim = 'storing/delayed_pretrain:' + str(pretune) + '_delay' + str(d) +'.csv'
    stims = pd.read_csv(f_stim)
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(' '+ str(d) + ' ms')
pac = plot_args(n_curves = len(dfs)+2, rb=False)
pa = plot_args(n_curves = len(dfs))
pa['ylim'] = [0, 1.1]
pa['colors'] = pac['colors'][:len(dfs)]
pa['labels'] = labels
pa['xspot'] = 3000
pa['right_margins'] = True
pa['plt_errors'] = True
pa['yadds'] = [0, 0, .02, -.03]
pa['title'] = 'Delayed Plasticity Pre-Trained (' + str(pretune) + ')'
pa['title_size'] = pa['label_size']
plot_pqs(dfs, pa)



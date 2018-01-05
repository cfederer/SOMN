"""Recreates Figure 3A from bioRxiv: 144683 (Federer & Zylberbeg 2017) """
from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *


ms = 3000
alphas = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 10]
n_iters = 10
############################# UNCOMMENT TO RUN NEW RESULTS #############################
'''
args = get_args()
args['ms'] = ms
args['n_iters'] = n_iters
args['tuned'] = True 

r = str(abs(np.round(np.random.randn(), 4))) 
total_time = 0

for i, alpha in enumerate(alphas):
    args['alpha'] = alpha
    print('Running alpha: ' + str(alpha))
    f_stim = ['storing/synaptic_noise_alpha=' + str(alpha) + '.csv']
    initialize_file(f_stim[0], args)
    t = PQ(args, stim_l = f_stim)
    print('Time to run  alpha=' + str(alpha) + ' ' + str(t/60) + ' minutes')
    total_time+=t
                
print('Total time to run ' + str(total_time/60) + ' minutes')
'''
dfs = list()
labels = list()

## add no noise
f_stim = 'storing/plastic_stim.csv'
stims = pd.read_csv(f_stim)
stims = pd.DataFrame.transpose(stims)
labels.append(' ' + r'$\alpha$ =  0')
dfs.append(stims[:ms])

## read in files 
for i, a in enumerate(alphas):
    f_stim = ['storing/synaptic_noise_alpha=' + str(a) + '.csv']
    stims = pd.read_csv(f_stim[0])
    stims = pd.DataFrame.transpose(stims)
    dfs.append(stims)
    labels.append(' ' + r'$\alpha$ =  ' + str(a))

### plot
pa = plot_args(n_curves = len(dfs),rb=False)
pa['labels'] =labels
pa['plt_errors'] = True
pa['xspot'] = 3000.5
t = -.2
b = -.63
g = .06
m = (t + b) / 2
pa['yadds'] = [t, t-g, t-2*g,t-3*g, t-4*g, t-5*g, t-6*g, t-7*g, t-8*g, t-9*g, t-10*g, 0]
pa['right_margins'] = True
plot_pqs_cutaxis(dfs, pa, [.992, 1.0001], [-.1, .3])


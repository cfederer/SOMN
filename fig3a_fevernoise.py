from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *

ms = 3000
sd = .00001
seed = 3
############################# UNCOMMENT TO RUN NEW RESULTS #############################
'''
args_f = get_args()
args_f['tuned'] = True
args_f['FEVER'] = True 
args_f['ms'] = ms
args_f['n_neurons'] = 100
args_f['noise_sd'] = sd
args_f['seed'] = seed 
nnf = NN(args_f)
Simf = Sim(nnf, args_f)
Simf.run()

args_t = get_args()
args_t['tuned'] = True
args_t['ms'] = ms
args_t['n_neurons'] = 100
args_t['noise_sd'] = sd
args_t['seed'] = seed
nnt = NN(args_t)
Simt = Sim(nnt, args_t)
Simt.run()

args_t['sdf'].to_csv('storing/Fig3A_tuned.csv')
args_f['sdf'].to_csv('storing/Fig3A_fever.csv')
'''
dft = pd.DataFrame.from_csv('storing/Fig3A_tuned.csv')
dff = pd.DataFrame.from_csv('storing/Fig3A_fever.csv')

## plot 
dfs = [dff, dft]
pa = plot_args()
pa['labels'] = ['FEVER + $\epsilon$', 'Plastic Random Synapse + $\epsilon$']
pa['colors'] = ['k', 'b']
pa['frac_stim'] = True
pa['xspot'] = 100
pa['yadds'] = [.3, -.5]
plot_dfs(dfs, pa)

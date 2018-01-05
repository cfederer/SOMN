from NN import * 
from qm import * 
import pandas as pd
import numpy as np
from plot_util import *

ms = 3000
n_iters = 10
############################# UNCOMMENT TO RUN NEW RESULTS #############################
'''
args_rw = get_args()
args_rw['tuned'] = True
args_rw['ms'] = ms
args_rw['n_iters'] = n_iters
args_rw['rws'] = True

args_b = get_args()
args_b['tuned'] = True
args_b['ms'] = ms
args_b['n_iters'] = n_iters
args_b['error'] = 'binary'
args_b['rws'] = True 

### create files for storing stim values 
r = str(abs(np.round(np.random.randn(), 4)))
b_stim = ['storing/binary_random.csv']
initialize_file(b_stim[0], args_b)
rw_stim = ['storing/random_weights.csv']
initialize_file(rw_stim[0], args_rw)

## run networks in parallel
print('Running random feedback network')
rt = PQ(args_rw, stim_l = rw_stim)
print('Running binary and random feedback networks')
bt = PQ(args_b, stim_l = b_stim)

print('Time to run tuned ' + str(rt/60) + ' minutes')
print('Time to run binary ' + str(bt/60) + ' minutes')
'''

############################# PLOTTING STORED RESULTS #############################

b_stim = ['storing/binary_random.csv'] ##comment out to run new results 
rw_stim = ['storing/random_weights.csv'] ##comment out to run new results 
u_stim = ['storing/constant_stim.csv'] ##comment out to run new results 

## read in files
binary_stims = pd.read_csv(b_stim[0])
binary_stims = pd.DataFrame.transpose(binary_stims)
rw_stims = pd.read_csv(rw_stim[0])
rw_stims = pd.DataFrame.transpose(rw_stims)
constant_stims = pd.read_csv(u_stim[0])
constant_stims = pd.DataFrame.transpose(constant_stims)

## plot
pa = plot_args(n_curves=3)
pa['labels'] =  ['Constant Synapse', 'Plastic (Binary & Random Feedback)', 'Plastic (Random Feedback)']
pa['colors']=['r', 'g', 'b']
pa['show'] = True
pa['plt_errors'] = True
pa['text_size'] = 16.5
pa['xspots'] = [150, 45, 45]
pa['ylim'] = [-.1, 1.15]
pa['yadds'] = [.01, -.18, .05] 
dfs = [constant_stims[:ms], binary_stims, rw_stims]
plot_pqs(dfs, pa)

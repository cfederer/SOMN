from NeuralNetwork import NN
from NetworkSimulation import Sim
from quantify_sims_multithreading import * 
import pandas as pd
import numpy as np
from arg_parsing import *
from plot_util import *

ms = 3000
n_iters = 100
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

args_u = get_args()
args_u['tuned'] = False
args_u['ms'] = ms
args_u['n_iters'] = n_iters

### create files for storing stim values 
r = str(abs(np.round(np.random.randn(), 4)))
b_stim = ['storing/binary_random_' + r + '.csv']
initialize_file(b_stim[0], args_b)
rw_stim = ['storing/randomweights_' + r + '.csv']
initialize_file(rw_stim[0], args_rw)
u_stim = ['storing/constant_stim ' + r + '.csv']
initialize_file(u_stim[0], args_u)

## run networks in parallel
print('Running random feedback network')
rt = PQ(args_rw, stim_l = rw_stim)
print('Running binary and random feedback networks')
bt = PQ(args_b, stim_l = b_stim)
print('Running constant network')
ut = PQ(args_u, stim_l = u_stim)

print('Time to run tuned ' + str(rt/60) + ' minutes')
print('Time to run binary ' + str(bt/60) + ' minutes')
print('Time to run untuned ' + str(ut/60) + ' minutes')
'''

############################# PLOTTING STORED RESULTS #############################

b_stim = ['storing/binary_random.csv'] ##comment out to run new results 
rw_stim = ['storing/randomweights.csv'] ##comment out to run new results 
u_stim = ['storing/constant_stim.csv'] ##comment out to run new results 

## read in files
binary_stims = pd.read_csv(b_stim[0])
binary_stims = pd.DataFrame.transpose(binary_stims)
rw_stims = pd.read_csv(rw_stim[0])
rw_stims = pd.DataFrame.transpose(rw_stims)
constant_stims = pd.read_csv(u_stim[0])
constant_stims = pd.DataFrame.transpose(constant_stims)

## plot
pa = plot_args()
pa['labels'] =  ['Constant Synapse', 'Plastic Synapse (Binary & Random Feedback)', 'Plastic Synapse (Random Feedback)']
pa['colors']=['red', 'green', 'blue']
pa['error_colors'] = ['#ff8080', '#c1f0c1', '#8080ff']
pa['save'] = True
pa['show'] = True
pa['save_as'] = 'binary_fdback_rws'
pa['plt_errors'] = True
pa['text_size'] = 14
pa['xspot'] = 200
pa['ylim'] = [-.2, 1.1]
pa['yadds'] = [-.09, -.06, -.09] 
dfs = [constant_stims, binary_stims, rw_stims]
plot_pqs(dfs, pa)


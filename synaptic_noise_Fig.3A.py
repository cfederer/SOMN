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
args_f = get_args()
args_f['tuned'] = True
args_f['FEVER'] = True 
args_f['ms'] = ms
args_f['n_iters'] = n_iters
args_f['n_neurons'] = 100
args_f['L_noise'] = .00001

args_t = get_args()
args_t['tuned'] = True
args_t['ms'] = ms
args_t['n_iters'] = n_iters
args_t['n_neurons'] = 100
args_t['noise_amp'] = 'alpha'
args_t['L_noise'] = .00001

r = str(abs(np.round(np.random.randn(), 4)))
f_stim = ['storing/FEVER_noise_' + str(args_f['L_noise']) + '_' +  r + '.csv']
initialize_file(f_stim[0], args_f)
tf = PQ(args_f, stim_l = f_stim)

t_stim = ['storing/plastic_noise_' + str(args_t['L_noise']) + '_' +  r + '.csv']
initialize_file(t_stim[0], args_t)
tt = PQ(args_t, stim_l = t_stim)
'''
############################# PLOTTING STORED RESULTS #############################

## read in files
f_stim = ['storing/FEVER_noise_1e-05.csv']  ## comment out to plot new results
stims_f = pd.read_csv(f_stim[0])
stims_f = pd.DataFrame.transpose(stims_f)

t_stim = ['storing/plastic_noise_1e-05.csv'] ## comment out to plot new results
stims_t = pd.read_csv(t_stim[0])
stims_t = pd.DataFrame.transpose(stims_t)

## plot 
dfs = [stims_f, stims_t]
pa = plot_args()
pa['labels'] = ['FEVER', 'Plastic Synapse']
pa['colors'] = ['black', 'blue']
pa['plt_errors'] = False
pa['error_colors'] = ['gray', '#8080ff']
pa['ylim'] = [-1, 10]
pa['text_size'] = 15
pa['xspot'] = 1430
pa['yadds'] = [-5, .2]
plot_pqs(dfs, pa)

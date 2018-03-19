import matplotlib.pyplot as plt
from NN_LIF import *
from qm import * 
from plot_util import *

ms = 3000
n_iters = 1
############## UNCOMMENT TO RUN NEW RESULTS #############

argst = get_args()
argst['ms'] = ms
argst['tuned'] = True
argst['n_iters'] = n_iters

argsu = get_args()
argsu['ms'] = ms
argsu['tuned'] = False
argsu['n_iters'] = n_iters

argst['steps'] = get_steps(argst)
argsu['steps'] = get_steps(argsu)

## create files
t_stim = ['storing/plastic_stim_LIF.csv']
initialize_file(t_stim[0], argst)
u_stim = ['storing/constant_stim_LIF.csv']
initialize_file(u_stim[0], argsu)

## run parallel 
print('tuned')
tt = PQ(argst, stim_l = t_stim)
print('untuned')
ut = PQ(argsu, stim_l = u_stim)

############## PLOT STORED RESULTS #############
t_stim = ['storing/plastic_stim_LIF.csv']
u_stim = ['storing/constant_stim_LIF.csv']
tuned = pd.read_csv(t_stim[0])
tuned_df = pd.DataFrame.transpose(tuned)
constant = pd.read_csv(u_stim[0])
constant_df = pd.DataFrame.transpose(constant)

## plot
dfs = [tuned_df, constant_df]
pa = plot_args(rb=False)
pa['labels'] = ['Plastic Random Synapse', 'Constant Random Synapse']
pa['plt_errors'] = True
pa['yadds'] = [-.1, -.15]
pa['xspot'] = 10
plot_pqs_LIF(dfs, pa, ms)

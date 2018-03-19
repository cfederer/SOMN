"""Recreates Figure 2A from bioRxiv: 144683 (Federer & Zylberbeg 2017)"""
from NN import * 
import pandas as pd
from plot_util import *

ms = 500
g = 2
##################### UNCOMMENT TO RUN NEW RESULTS ###################
'''
## run constant random synapse network
args_u = get_args()
args_u['tuned'] = False
args_u['seed'] = get_paper_seed()
args_u['store_frs'] = True
args_u['ms'] = ms
args_u['g'] = g
NNu = NN(args_u) ## the neural network object
Simu = Sim(NNu, args_u) ## creates a simulation 
Simu.run() ## runs the simulation 
args_u['sdf'].to_csv('storing/Fig2A_untuned_chaos' + str(g) + '.csv')
args_u['frs'].to_csv('storing/Fig2A_untunedfrs_chaos' + str(g) + '.csv')
print('Running plastic synapse network')

## run plastic random synapse network 
args_t = get_args()
args_t['tuned'] = True
args_t['seed'] = get_paper_seed()
args_t['store_frs'] = True
args_t['ms'] = ms
args_t['hist'] = True
args_t['g'] = g
NNt = NN(args_t)
Simt=Sim(NNt, args_t)
Simt.run()
args_t['sdf'].to_csv('storing/Fig2A_tuned_chaos' + str(g) + '.csv')
args_t['frs'].to_csv('storing/Fig2A_tunedfrs_chaos' + str(g) + '.csv')
'''
##################### RUN STORED RESULTS ###################
dft = pd.DataFrame.from_csv('storing/Fig2A_tuned_chaos' + str(g) + '.csv')
dfu = pd.DataFrame.from_csv('storing/Fig2A_untuned_chaos' + str(g) + '.csv')
frst = pd.DataFrame.from_csv('storing/Fig2A_tunedfrs_chaos' + str(g) + '.csv')
frsu = pd.DataFrame.from_csv('storing/Fig2A_untunedfrs_chaos' + str(g) + '.csv')

pa = plot_args(poster=True)
pa['text_size'] = 19
pa['ylabel'] = ''
pa['save_as'] = 'fig2a_chaos'
pa['save'] = True
## recreates figure 2A 
plot_activities_chaos(dft, dfu, frst, frsu, pa)


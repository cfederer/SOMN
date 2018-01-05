"""Recreates Figure 2A from bioRxiv: 144683 (Federer & Zylberbeg 2017)"""
from NN import * 
import pandas as pd
from plot_util import *

ms = 500
##################### UNCOMMENT TO RUN NEW RESULTS ###################
'''
## run constant random synapse network
args_u = get_args()
args_u['tuned'] = False
args_u['seed'] = get_paper_seed()
args_u['store_frs'] = True
args_u['ms'] = ms 
NNu = NN(args_u) ## the neural network object
Simu = Sim(NNu, args_u) ## creates a simulation 
Simu.run() ## runs the simulation 
args_u['sdf'].to_csv('storing/Fig2A_untuned.csv')
args_u['frs'].to_csv('storing/Fig2A_untunedfrs.csv')
print('Running plastic synapse network')

## run plastic random synapse network 
args_t = get_args()
args_t['tuned'] = True
args_t['seed'] = get_paper_seed()
args_t['store_frs'] = True
args_t['ms'] = ms
args_t['hist'] = True
NNt = NN(args_t)
Simt=Sim(NNt, args_t)
Simt.run()
args_t['sdf'].to_csv('storing/Fig2A_tuned.csv')
args_t['frs'].to_csv('storing/Fig2A_tunedfrs.csv')
'''
##################### RUN STORED RESULTS ###################
dft = pd.DataFrame.from_csv('storing/Fig2A_tuned.csv')
dfu = pd.DataFrame.from_csv('storing/Fig2A_untuned.csv')
frst = pd.DataFrame.from_csv('storing/Fig2A_tunedfrs.csv')
frsu = pd.DataFrame.from_csv('storing/Fig2A_untunedfrs.csv')

pa = plot_args()
pa['text_size'] = 19
pa['ylabel'] = ''
## recreates figure 2A 
plot_activities(dft, dfu, frst, frsu, pa)


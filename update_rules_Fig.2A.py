"""Recreates Figure 2A from bioRxiv: 144683 (Federer & Zylberbeg 2017)
as well as the remembered stimulus values for the network for 3 s"""
from NeuralNetwork import NN
from NetworkSimulation import Sim
import pandas as pd
from plot_util import *
from arg_parsing import *

ms = 3000 ## ms to run trial for 

print('Running constant synapse network')
## run constant network 
args_u = get_args()
args_u['tuned'] = False
args_u['seed'] = get_paper_seed()
args_u['store_frs'] = True
args_u['ms'] = ms 
NNu = NN(args_u) ## the neural network object
Simu = Sim(NNu, args_u) ## creates a simulation 
Simu.run() ## runs the simulation 

print('Running plastic synapse network')
## run plastic network 
args_t = get_args()
args_t['tuned'] = True
args_t['seed'] = get_paper_seed()
args_t['store_frs'] = True
args_t['ms'] = ms 
NNt = NN(args_t)
Simt=Sim(NNt, args_t)
args_tuned = Simt.run()

##plot 
args = [args_u, args_t]
pa = plot_args()
pa['text_size'] = 15
## recreates figure 2A 
plot_activities(args_t, args_u, pa)
## plots remembered stimulus values only 
pa['labels'] = ['Constant Synapse', 'Plastic Synapse']
pa['colors'] = ['red', 'blue']
pa['ylim'] = [-1, 31]
pa['ylabel'] = 'Remembered Stimulus Value'
pa['right_margins'] = True
plot_stims(args, pa) 

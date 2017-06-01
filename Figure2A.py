"""Recreates Figure 2A from bioRxiv: 144683 (Federer & Zylberbeg 2017)
  as well as the remembered stimulus values for the network for 3 s"""
from NeuralNetwork import NN
from NetworkSimulation import Sim
from the_generic_plotter import plot_stims, plot_activities
import pandas as pd
from arg_parsing import get_args

### command line arguments, all defaults set otherwise 
args = get_args()

def initialize_file(loc, cols):
    """ initializes a file with column names """
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()

print('Running constant synapse network')
##untuned
NNu = NN(n_neurons=args.n_neurons, seed=args.seed, activation=args.activation, n_stim=args.n_stim)
Simu = Sim(ms=args.ms, NNet=NNu, dt=args.dt, eta=args.eta, update_rs = True, n_rs = args.nrs)
Simu.run()

print('Running plastic synapse network')
##tuned 
NNt = NN(n_neurons=args.n_neurons, plastic_synapse=True, seed=args.seed, activation=args.activation, n_stim=args.n_stim)
Simt=Sim(ms=args.ms, NNet=NNt, dt=args.dt, eta=args.eta, update_rs = True, n_rs = args.nrs)
Simt.run()

##plot 
dfs = [Simu.sdf, Simt.sdf]
labels = ['Constant Synapse', 'Plastic Synapse']
plot_activities(stimsu=Simu.sdf, actsu=Simu.frs[:51], stimst=Simt.sdf, actst=Simt.frs[:51], dt=args.dt)
plot_stims(dfs, labels, dt=args.dt, ylabel='Remembered Stimulus Value',
           plt_errors=False, colors=['red', 'blue'], yadds = [1, -2], ylim = [-1, 31])

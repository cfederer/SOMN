from NeuralNetwork import NN
from NetworkSimulation import Sim
from the_generic_plotter import plot_stims, plot_activities
import pandas as pd

def initialize_file(loc, cols):
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()

def write_to_file(loc, x):
    f =  open(loc, 'a+')
    f.write(str(x))
    f.write('\n')
    f.close()

ms = 3001
seed= 19
n_neurons=100
dt = .001
eta = .0001
nrs = 10 
activation = 'relu'
n_stim = 1

print('Running constant synapse network')
##untuned
NNu = NN(n_neurons=n_neurons, seed=seed, activation=activation, n_stim=n_stim)
Simu = Sim(ms=ms, NNet=NNu, dt=dt, eta=eta, update_rs = True, n_rs = nrs)
Simu.run()

print('Running plastic synapse network')
##tuned 
NNt = NN(n_neurons=n_neurons, plastic_synapse=True, seed=seed, activation=activation, n_stim=n_stim)
Simt=Sim(ms=ms, NNet=NNt, dt=dt, eta=eta, update_rs = True, n_rs = nrs)
Simt.run()

save= False 

dfs = [Simu.sdf, Simt.sdf]
labels = ['Constant Synapse', 'Plastic Synapse']

#plot_activities(Simu.sdf[:51], Simu.frs[:51], Simt.sdf[:51], Simt.frs[:51], dt=dt)
plot_activities(stimsu=Simu.sdf, actsu=Simu.frs[:51], stimst=Simt.sdf, actst=Simt.frs[:51], dt=dt)
plot_stims(dfs, labels, dt=.001, ylabel='Remembered Stimulus Value',
           plt_errors=False, colors=['red', 'blue'], yadds = [1, -2], ylim = [-1, 31])

from NeuralNetwork import NN
from NetworkSimulation import Sim
import sys
sys.path.insert(0, 'plotting/')
'''
from plot_activities import plot_activities
from plot_one_stim import plot_stims
'''
from the_generic_plotter import plot_multidim


save= False
ms = 50
seed= 19
n_neurons=100
dt = .001
eta = .0001
activation = 'relu'
n_stim = 10

##untuned
NNu = NN(n_neurons=n_neurons, seed=seed, activation=activation, n_stim=n_stim)
Simu = Sim(ms=ms, NNet=NNu, dt=dt, eta=eta, updates=False)
Simu.run()

##tuned 
NNt = NN(n_neurons=n_neurons, plastic_synapse=True, seed=seed, activation=activation, n_stim=n_stim)
Simt=Sim(ms=ms, NNet=NNt, dt=dt, eta=eta, updates=False)
Simt.run()




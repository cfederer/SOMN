from NeuralNetwork import NN
from NetworkSimulation import Sim
import pandas as pd
import numpy as np
import quantify_sims_multithreading as pq
from scipy import stats

n_neurons = 100
ms = 100
n_iters = 10
dt = .001
eta = .0001

NNt = NN(n_neurons = n_neurons, plastic_synapse=True)
Simt = Sim(NNet=NNt, ms=ms)
print('Made Simt')
Sim2 = Sim(Sim_copy = Simt)


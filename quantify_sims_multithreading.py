import multiprocessing
from multiprocessing import Process, Queue 
from NeuralNetwork import NN
from NetworkSimulation import Sim
import numpy as np
import pandas as pd
import time

import warnings
warnings.filterwarnings("ignore")

def avg(x):
    return sum(x) / len(x)

def write_to_file(loc, x):
    f =  open(loc, 'a+')
    f.write(str(x))
    f.write('\n')
    f.close()

def trained_network(n, seed, n_neurons, ms, eta, dt, plastic_synapse):
    NNx = NN(n_neurons = n_neurons, plastic_synapse=True)
    Simx = Sim(NNet=NNx, ms=ms, eta=eta, dt=dt, updates=False)
    Simx.run()
    for i in range(n-1):
        NNx = NN(n_neurons=n_neurons, plastic_synapse=True, d = NNx.d, L=NNx.L)
        Simx = Sim(NNet=NNx, ms=ms, eta=eta, dt=dt, updates=False)
        Simx.run()
    return NN(n_neurons=n_neurons, plastic_synapse=plastic_synapse, d = NNx.d, L = NNx.L, seed=seed)

def quant(plastic_synapse, activation,n_stim, dt, eta, frac_tuned, n_neurons, ms, sd, L_noise, rws, pretune, connectivity):
    """ runs a Sim, quantifies fraction of stim retained and returns """
    if(pretune==0):
        NNx = NN(n_neurons=n_neurons,plastic_synapse=plastic_synapse,n_stim=n_stim,activation=activation, frac_tuned=frac_tuned,
                      L_noise=L_noise, rws=rws, seed=sd, connectivity=connectivity)
    else:
        NNx = trained_network(pretune, sd, n_neurons, ms, eta, dt, plastic_synapse)
    Simx = Sim(ms=ms, NNet=NNx, updates=False, dt=dt)
    Simx.run()
    if(n_stim ==1):
        stim = (Simx.sdf / float(Simx.initial_stim)).values.ravel()
    else:
        stim = list()
        for i in range(n_stim):
            stim.append(Simx.sdf[i] / float(Simx.initial_stim[0][i]))
    return stim

def worker(iters, stim_lock, stim_l,  n, plastic_synapse, activation, n_stim, dt, eta, frac_tuned, n_neurons,
           ms, sd, L_noise, rws, pretune, connectivity):
    """ worker function takes Sims to run and a queue to place lists of
        fraction of stim retained """
    for i in range(len(iters)):
        print('Running worker ' + str(n) + ' Sim ' + str(i))
        stim = quant(plastic_synapse, activation,n_stim, dt, eta, frac_tuned, n_neurons, ms, sd, L_noise, rws, pretune, connectivity)
        if(n_stim == 1):
            stim = [stim]
        for i in range(n_stim):
            stim_csv = ','.join(['%.5f' % num for num in stim[i]])
            stim_lock.acquire()
            write_to_file(stim_l[i], stim_csv)
            stim_lock.release()
            
def PQ(n_iters,n_neurons, stim_l, ms,plastic_synapse, activation='relu', n_stim=1, dt=.01,
       eta=.0001, seeds=None, frac_tuned=1, nps=None, L_noise = 0, rws = False, pretune=0, connectivity = 1):

    start_time = time.time()
    """ splits up Sims into num of available cpus - 1, runs and puts list of
        results on queue """
    if nps is None:
        nps = 4 
    if n_iters < nps:
        nps = n_iters 
    print('Running ' + str(nps) + ' processes')
    
    stim_lock = multiprocessing.Lock()
    iterchunks = np.array_split(list(range(n_iters)), nps)
    procs = list() 
    ## start process 
    for i in range(nps):
        if seeds is not None:
            sd = seeds[i]
        else:
            sd = np.random.randint(0,1000)
        p = multiprocessing.Process(
            target=worker,
            args = (iterchunks[i], stim_lock, stim_l, i, plastic_synapse, activation,
                    n_stim, dt, eta, frac_tuned, 
                    n_neurons, ms, sd, L_noise, rws, pretune, connectivity))
        procs.append(p)
        p.start()
    
    for p in procs:
        p.join()
    return (time.time() - start_time)



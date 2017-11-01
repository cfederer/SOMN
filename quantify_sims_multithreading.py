""" Code to run randomly initialized network in parallel"""

import multiprocessing
from multiprocessing import Process, Queue 
from NeuralNetwork import NN
from NetworkSimulation import Sim
import numpy as np
import pandas as pd
import time
import copy

import warnings
warnings.filterwarnings("ignore")

def initialize_file(loc, args):
    """ Initializes a file with column names for saving"""
    cols = list(range(args['ms']))
    cols = ','.join(['%.5f' % num for num in cols])
    f = open(loc, 'a+')
    f.write(cols+'\n')
    f.close()

def write_to_file(loc, x):
    """ Write x to file at spot loc """
    f =  open(loc, 'a+')
    f.write(str(x))
    f.write('\n')
    f.close()

def train_network(args):
    """ returns a pre-trained network """
    train_args = copy.deepcopy(args)
    train_args['tuned'] = True
    
    NNx = NN(train_args) 
    Simx = Sim(NNx, train_args)
    Simx.run()
    for i in range(train_args['pretune']-1):
        NNx = NN(train_args)
        Simx = Sim(NNx, train_args)
        Simx.run()
    train_args['tuned'] = args['tuned']
    return train_args 

def quant(args, sd):
    """ runs a Sim and returns fraction of stim retained """
    args['seed'] = sd 
    if(args['pretune']==0):
        NNx = NN(args)
    else:
        args = train_network(args)
        NNx = NN(args)
        
    Simx = Sim(NNx, args)
    Simx.run()
    if(args['n_stim'] ==1):
        stim = (args['sdf'] / float(args['initial_stim'])).values.ravel()
    else:
        stim = list()
        for i in range(args['n_stim']):
            stim.append(args['sdf'][i] / float(args['initial_stim'][0][i]))
    return stim

def worker(iters, stim_lock, stim_l, args, sd, n):
    """ worker function takes Sims to run and a queue to place lists of
        fraction of stim retained """
    for i in range(len(iters)):
        print('Running worker ' + str(n) + ' Sim ' + str(i))
        stim = quant(copy.deepcopy(args), sd)
        if(args['n_stim'] == 1):
            stim = [stim]
        for i in range(args['n_stim']):
            stim_csv = ','.join(['%.5f' % num for num in stim[i]])
            stim_lock.acquire()
            write_to_file(stim_l[i], stim_csv)
            stim_lock.release()
            
def PQ(args, stim_l, nps=None):
    start_time = time.time()
    """ splits up Sims into num of available cpus - 1, runs and puts list of
        results on queue """
    if nps is None:
        nps = 4 
    if args['n_iters'] < nps:
        nps = args['n_iters']
    print('Running ' + str(nps) + ' processes')
    
    stim_lock = multiprocessing.Lock()
    iterchunks = np.array_split(list(range(args['n_iters'])), nps)
    procs = list() 
    ## start process 
    for i in range(nps):
        sd = np.random.randint(0,1000)
        p = multiprocessing.Process(
            target=worker,
            args = (iterchunks[i], stim_lock, stim_l, args, sd, i))
        procs.append(p)
        p.start()
    
    for p in procs:
        p.join()
    return (time.time() - start_time)



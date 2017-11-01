""" helper code for default arguments for running different network simulations"""

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def avg(x):
    """ Returns avg"""
    return sum(x) / len(x)

def get_args():
    """ default arguments for a network simulation """
    args_dict = dict()
    args_dict['error'] = 'derived'                       ## error is (ds/dt)^2; other option is 'binary' for sign(dsdt)
    args_dict['tuned'] = True                            ## plastic random synapses, False for constant random synapses
    args_dict['ms'] = 3001                               ## ms to run simulation for
    args_dict['n_neurons'] = 100                         ## num neurons in the network 
    args_dict['dt'] = .001                               ## the time scale
    args_dict['eta'] = .0001                             ## learning rate
    args_dict['activation'] = 'relu'                     ## activation function, other option is 'linear'
    args_dict['n_stim'] = 1                              ## the number of stim to be stored 
    args_dict['nrs'] = 10                                ## number of firing rates to plot for Fig 2A
    args_dict['seed'] = np.random.randint(0,1000)        ## needs associated seed for parallelizing
    args_dict['n_iters'] = 100                           ## number of iterations to run 
    args_dict['abs_d'] = True                            ## abs values of read-out, d
    args_dict['tau'] = 1                                 ## time scale 
    args_dict['store_frs'] = False                       ## store firing rates for Fig 2A
    args_dict['frac_tuned'] = 1                          ## fraction of plastic synapses
    args_dict['connectivity'] = 1                        ## connection probability
    args_dict['FEVER'] = False                           ## runs FEVER like entwork (Druckmann & Chklovskii Curr Biol 2012)
    args_dict['updates'] = False                         ## get printed updates while running 
    args_dict['L_noise'] = 0                             ## amplitude of synaptic noise 
    args_dict['noise_amp'] = 'update'                    ## std of Gaussian noise update size, other option 'alpha'
    args_dict['rws'] = False                             ## random feedback weights, d == q, if False: d != q
    args_dict['pretune'] = 0                             ## number of times to train on previous stim
    return args_dict 


def get_paper_seed():
    """ Seed used in paper to exactly recreate Figure 2A """
    return 19 

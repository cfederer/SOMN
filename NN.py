""" A neural network object """
import numpy as np
import math
from scipy import linalg
from random import randint
import copy
import math
import pandas as pd 

def get_args():
    """ default arguments for a network simulation """
    args_dict = dict()
    args_dict['g'] = 1                                   ## chaotic > 1
    args_dict['error'] = 'derived'                       ## error is (ds/dt)^2; other option is 'binary' for sign(dsdt)
    args_dict['tuned'] = True                            ## plastic random synapses, False for constant random synapses
    args_dict['ms'] = 3001                               ## ms to run simulation for
    args_dict['n_neurons'] = 100                         ## num neurons in the network 
    args_dict['dt'] = .001                               ## the time scale
    args_dict['eta'] = .0001                             ## learning rate
    args_dict['activation'] = 'relu'                     ## activation function, other option is 'linear'
    args_dict['n_stim'] = 1                              ## the number of stim to be stored 
    args_dict['nrs'] = 10                                ## number of firing rates to plot for Fig 2A
    args_dict['seed'] = np.random.randint(0,10000)       ## needs associated seed for parallelizing
    args_dict['abs_d'] = True                            ## abs values of read-out, d
    args_dict['tau'] = 1                                 ## time scale 
    args_dict['store_frs'] = False                       ## store firing rates for Fig 2A
    args_dict['store_drs'] = False 
    args_dict['frac_tuned'] = 1                          ## fraction of plastic synapses
    args_dict['connectivity'] = 1                        ## connection probability
    args_dict['FEVER'] = False                           ## runs FEVER like entwork (Druckmann & Chklovskii Curr Biol 2012)
    args_dict['updates'] = False                         ## get printed updates while running 
    args_dict['alpha'] = 0                               ## amplitude of synaptic noise
    args_dict['v_alpha'] = 0                             ## amplitude of voltage / activity noise 
    args_dict['rws'] = False                             ## random feedback weights, d == q, if False: d != q
    args_dict['pretune'] = 0                             ## number of previous stim to train on 
    args_dict['n_iters'] = 10                            ## number of times to run if quantifying remembered value relative to initial 
    args_dict['delay'] = False                           ## add delay for error signal to reach synaptic update s
    args_dict['delay_ms'] = 1                            ## ms of delay 
    return args_dict 


def get_paper_seed():
    """ Seed used in paper to exactly recreate Figure 2A """
    return 19 

def avg(x):
    """ Returns the average of x """
    x2 = copy.deepcopy(x)
    return sum(x2) / len(x2)

def vect_pinv(v):
    """ Takes the Moore-Penrose pseudoinverse of a vector v"""
    vT = np.transpose(v)
    vTv = np.dot(vT, v)
    vpinv = np.true_divide(vT, vTv)
    return(vpinv)

def binary_updates(x):
    """ returns sign(x) """
    if(x < 0):
        return -1
    if(x>0):
        return 1
    return 0 

def apply_mask(the_mask, the_matrix):
    """ Multiples a mask matrix to the matrix to be masked """
    return np.multiply(the_matrix, the_mask)

def mask_matrix(num_cells, frac_tuned):
    """ Creates a mask matrix """
    x = np.random.uniform(0,1,size=(num_cells,num_cells))
    y = np.zeros((num_cells,num_cells))
    y[x<=frac_tuned] = 1
    return y

def relu(x):
    """ Return x if x>0, else 0"""
    x2 = copy.deepcopy(x)
    return np.maximum(0,x2)

def d_relu(x):
    """ Return 1 if x >0, else 0 """
    x2 = copy.deepcopy(x)
    x2[x2>0] = 1
    x2[x2<0] = 0
    return x2

class NN(object):
    def __init__(self, args):
        """ Return a new NN object from specified parameters """
        np.random.seed(args['seed'])
        if(args['FEVER']):
            args['activation'] = ['linear']
        args['a'] = np.random.randn(args['n_neurons'], 1)
        if(args['activation'] == 'relu'):
            args['r'] = relu(args['a'])
        else:
            args['r'] = args['a']
        if 'd' not in args:
            if(args['abs_d']):
                args['d'] = abs(np.random.randn(args['n_neurons'], args['n_stim']))
            else: 
                args['d'] = np.random.randn(args['n_neurons'], args['n_stim'])
        if(args['rws']):
            if(args['abs_d']):
                args['q'] = abs(np.random.randn(args['n_neurons'], args['n_stim']))
            else: 
                args['q'] = np.random.randn(args['n_neurons'], args['n_stim'])
        else:
            args['q'] = args['d']
            
        if(args['FEVER']):
            dpinv = vect_pinv(args['d'])
            args['L'] = np.outer(args['d'], dpinv)
        else:
            if 'L' not in args:
                args['L'] = np.random.randn(args['n_neurons'], args['n_neurons']) * (args['g']/ math.sqrt(args['n_neurons']))
                np.fill_diagonal(args['L'], 0)
        if(args['connectivity'] != 1):
            args['connectivity_mask'] = mask_matrix(args['n_neurons'], args['connectivity'])
        if(args['frac_tuned'] != 1):
            args['mask'] = mask_matrix(args['n_neurons'], args['frac_tuned'])
        self.args = args
        
    def calc_s(self):
        """ Calculate and return the rememberd stim value (Equation 5) q==d unless rws==True """
        return np.dot(np.transpose(self.args['r']), self.args['d'])

    def update_a(self):
        """ Update activity of network (Equation 1) """
        if(self.args['v_alpha'] ==0):
            self.args['a'] = self.args['a']*(1-self.args['dt']) + self.args['dt']*(np.dot(self.args['L'], self.args['r']))
        else:
            noise = self.args['v_alpha'] * np.random.normal() * self.args['dt']*(np.dot(self.args['L'], self.args['r']))
            self.args['a'] = self.args['a']*(1-self.args['dt']) + self.args['dt']*(np.dot(self.args['L'], self.args['r'])) + noise 

    def update_r(self):
        """ Update firing rates of network """
        if(self.args['activation']=='relu'): 
            self.args['r'] = relu(self.args['a'])
        else: ##linear 
            self.args['r'] = self.args['a']

    def drdt(self):
        """ Calculate and return the derivative of the firing rates """
        if(self.args['activation']=='relu'):
            return d_relu(self.args['a'])
        else: ##linear 
            return 1 

    def calc_dsdt(self):
        """ Calculate and return change in stim """
        drdt = self.drdt()
        if(self.args['n_stim'] >1):
            dsdt = (np.transpose(np.dot(np.transpose(self.args['d']*drdt),  (-self.args['a'] + np.dot(self.args['L'],self.args['r'])))))[0]
        else:
            dsdt = np.vdot(self.args['d']*drdt, (-self.args['a'] + np.dot(self.args['L'],self.args['r'])))
        if(self.args['error'] == 'binary'):
            return(binary_updates(dsdt))
        return dsdt 
        
    def update_L(self):
        """ Update the connectivity matrix, L (Equation 3, 4) """
        drdt = self.drdt()
        dsdt = self.calc_dsdt()
        dL = (np.repeat(np.dot(self.args['q'],dsdt), self.args['n_neurons'])).reshape(self.args['n_neurons'],self.args['n_neurons']) 
        dL = dL * 2 * self.args['eta'] * np.transpose(self.args['r']) * drdt
        if(self.args['alpha'] != 0):
            noise = self.args['alpha'] * dL * np.random.normal()
            if(self.args['FEVER']):
                self.args['L'] += noise
            else: 
                dL += noise
        if(self.args['frac_tuned'] != 1):
            self.args['L'] = self.args['L'] - apply_mask(self.args['mask'], dL)
        else:
            self.args['L'] = self.args['L'] - dL 
        if(self.args['connectivity'] < 1):
            self.args['L'] = apply_mask(self.args['connectivity_mask'], self.args['L'])
        if(not self.args['FEVER']):
            np.fill_diagonal(self.args['L'], 0)

    def update_L_delayed(self, t):
        """ Update the connectivity matrix, L (Equation 3, 4) with delayed feedback"""
        drdt = self.drdt()
        feedback_t = t - (self.args['stepsperms'] * self.args['delay_ms'])
        if(feedback_t < 0):
            dsdt = -1
        else: 
            dsdt = self.args['feedback'].ix[feedback_t]
        dL = (np.repeat(np.dot(self.args['q'],dsdt), self.args['n_neurons'])).reshape(self.args['n_neurons'],self.args['n_neurons']) 
        dL = dL * 2 * self.args['eta'] * np.transpose(self.args['r']) * drdt
        if(self.args['alpha'] != 0):
            noise = self.args['alpha'] * dL * np.random.normal()
            if(self.args['FEVER']):
                self.args['L'] += noise
            else: 
                dL += noise
        if(self.args['frac_tuned'] != 1):
            self.args['L'] = self.args['L'] - apply_mask(self.args['mask'], dL)
        else:
            self.args['L'] = self.args['L'] - dL 
        if(self.args['connectivity'] < 1):
            self.args['L'] = apply_mask(self.args['connectivity_mask'], self.args['L'])
        if(not self.args['FEVER']):
            np.fill_diagonal(self.args['L'], 0)

class Sim(object):
    """ A simulation of a neural network.
        args: the arguments dictionary with values set 
    """

    def __init__(self, NNt, args):
        """ Returns a new NetworkSimulation object with specified parameters or copies parameters from Sim_copy """
        args['NNt'] = NNt
        args['steps'] = args['ms'] / (args['dt'] * args['tau'] * 20)
        args['stepsperms'] = 1 / (args['dt']*args['tau']*20) 
        if(args['store_frs']):
            args['frs'] = pd.DataFrame(np.zeros((args['ms'], args['nrs'])))
            args['frs_idx'] = np.random.choice(list(range(args['n_neurons'])), args['nrs'])
        if(args['store_drs']):
            args['drs'] = pd.DataFrame(np.zeros((args['ms'], args['n_neurons'])))
        args['initial_stim'] = args['NNt'].calc_s()
        args['sdf'] = pd.DataFrame(np.zeros((args['ms'], args['n_stim']))) ### stim values
        if(args['frac_tuned'] == 0):
            args['tuned'] = False  
        if(args['delay']):
            args['feedback'] = pd.DataFrame(np.zeros((int(args['steps']), args['n_stim'])))
        self.args = args
    
    def update_sdf(self, mst):
        """ Update remembered stimulus in dataframe sdf """
        s = self.args['NNt'].calc_s()
        for i in range(self.args['n_stim']):
            self.args['sdf'][i][mst] = s[0][i]
    
    def update_frs(self, mst):
        """ Update the neural activities in dataframe activities """
        frs = [r for sublist in self.args['r'][self.args['frs_idx']].tolist() for r in sublist]
        self.args['frs'].ix[mst] = frs

    def update_drs(self, mst):
        drs = self.args['NNt'].drdt().reshape(self.args['n_neurons'],)
        self.args['drs'].ix[mst] = drs 

    def update_feedback(self, t):
        """ stores feedback for delayed plasticity """
        self.args['feedback'].ix[t] = self.args['NNt'].calc_dsdt()

    def run(self):
        """ Run a network simulation """
        t = 0 
        while(t < self.args['steps']):
            if(t % self.args['stepsperms'] == 0):
                if(t == 0):
                    mst = 0
                else:
                    mst = int(t*self.args['dt']*self.args['tau']*20)
                self.update_sdf(mst) ## storing stim values 
                if(self.args['store_frs']): ## storing firing rates
                    self.update_frs(mst)
                if(self.args['updates'] and t % 5000 ==0): ## printing updates 
                    print(str(mst) +  ' ms ' +  str(self.args['NNt'].calc_s()))
                if(self.args['store_drs']):
                    self.update_drs(mst)
            if(self.args['delay']):
                self.update_feedback(t)
            self.args['NNt'].update_a()
            self.args['NNt'].update_r()
            if(self.args['tuned']):
                if(self.args['delay']):
                    self.args['NNt'].update_L_delayed(t)
                else:
                    self.args['NNt'].update_L()
            t += 1
        return self.args 
            


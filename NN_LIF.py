""" A neural network object """
import numpy as np
import math
from scipy import linalg
from random import randint
import copy
import math
import pandas as pd

def avg(x):
    """ Returns the average of x """
    x2 = copy.deepcopy(x)
    return sum(x2) / len(x2)

def relu(x):
    """ Return x if x>0, else 0"""
    x2 = copy.deepcopy(x)
    return np.maximum(0,x2)

def apply_mask(the_mask, the_matrix):
    """ Multiples a mask matrix to the matrix to be masked """
    return np.multiply(the_matrix, the_mask)

def mask_matrix(x, y, frac_tuned):
    """ Creates a mask matrix """
    st = np.random.uniform(0,1,size=(x,y))
    m = np.zeros((x,y))
    m[st<=frac_tuned] = 1
    return m

def get_args():
    """ default arguments for a network simulation """
    a = dict()
    a['tuned'] = True                            ## plastic random synapses, False for constant random synapses
    a['ms'] = 20                               ## ms to run simulation for
    a['vthr'] = -50
    a['vrest'] = -70
    a['vreset'] = -60 
    a['dt'] = .01
    a['tau'] = 1
    a['tauE'] = 1
    a['tauI'] = .5
    a['ne'] = 100
    a['ni'] = 20
    a['eta'] = 1000
    a['n_stim'] = 1
    a['pretune'] = 0
    a['frac_tuned'] = 1
    a['connectivity'] = 1 
    return a

def get_steps(args):
    return int(args['ms'] / (args['dt'] * args['tau'] * 20))

class NN(object):
    def __init__(self, args):
        ## timing
        args['steps'] = int(args['ms'] / (args['dt'] * args['tau'] * 20))
        args['stepsperms'] = 1 / (args['dt']*args['tau']*20)
        args['mspersteps'] = args['dt']*args['tau']*20
        ## network vars
        if 'seed' in args:
            np.random.seed(args['seed'])
        args['ve'] = np.random.randint(args['vreset'], args['vthr']+10, size=(args['ne']))
        args['vi'] = np.random.randint(args['vreset'], args['vthr']+10, size=(args['ni']))
        args['d'] = abs(np.random.randn(args['ne'], 1))
        args['lee'] = relu(np.random.randn(args['ne'], args['ne']))
        np.fill_diagonal(args['lee'], 0)
        args['lii'] = relu(np.random.randn(args['ni'], args['ni']))
        np.fill_diagonal(args['lii'], 0)
        args['lei'] = relu(np.random.randn(args['ne'], args['ni'])) / math.sqrt(args['ne'])
        np.fill_diagonal(args['lei'], 0)
        args['lie'] = relu(np.random.randn(args['ni'], args['ne'])) / math.sqrt(args['ni'])
        np.fill_diagonal(args['lie'], 0)
        args['mee'] = np.zeros((args['ne'], args['ne']))
        args['mii'] = np.zeros((args['ni'], args['ni']))
        args['mei'] = np.zeros((args['ne'], args['ni'])) 
        args['mie'] = np.zeros((args['ni'], args['ne']))
        args['xee'] = np.zeros((args['ne'], args['ne']))
        args['xii'] = np.zeros((args['ni'], args['ni']))
        args['xei'] = np.zeros((args['ne'], args['ni'])) 
        args['xie'] = np.zeros((args['ni'], args['ne']))
        ##robustness
        if(args['frac_tuned'] != 1 and args['frac_tuned'] !=0):
            args['mask'] = mask_matrix(args['ne'], args['ne'], args['frac_tuned'])
        elif(args['frac_tuned'] == 0):
            args['tuned'] = False
        if(args['connectivity'] != 1):
            args['mask_lee'] = mask_matrix(args['ne'], args['ne'], args['connectivity'])
            args['mask_lii'] = mask_matrix(args['ni'], args['ni'], args['connectivity'])
            args['mask_lei'] = mask_matrix(args['ne'], args['ni'], args['connectivity'])
            args['mask_lie'] = mask_matrix(args['ni'], args['ne'], args['connectivity'])
            ## apply masks 
            args['lee'] = apply_mask(args['mask_lee'], args['lee'])
            args['lii'] = apply_mask(args['mask_lii'], args['lii'])
            args['lei'] = apply_mask(args['mask_lei'], args['lei'])
            args['lie'] = apply_mask(args['mask_lie'], args['lie'])
        ##initial stim representation
        self.args = args
        self.update_spikes()
        args['re'] = self.args['spikes_e'] 
        args['s_initial'] = np.dot(args['spikes_e'], args['d'])[0]
        args['s'] = args['s_initial']
        self.args = args
        
    def update_spikes(self):
        self.args['spikes_e'] = np.array([self.args['ve'] >= self.args['vthr']]).astype(int)
        self.args['spikes_i'] = np.array([self.args['vi'] >= self.args['vthr']]).astype(int)

    def update_ds(self):
        self.args['ds'] = (self.args['dt'] / self.args['tau']) * (-self.args['s'] + np.dot(self.args['spikes_e'], self.args['d']))

    
    def update_s(self):
        self.args['s'] = self.args['s'] + self.args['ds']
 
    def update_ve(self):
        #print('update ve')
        dve = (self.args['dt'] / self.args['tauE']) * ((self.args['vrest'] - self.args['ve']) +
                np.reshape(np.dot(sum(self.args['mee']), self.args['lee']), self.args['ne'],) -
                np.reshape(np.dot(sum(self.args['mei']), np.transpose(self.args['lei'])), self.args['ne'],)
                + abs(np.random.randn(self.args['ne'])))
        #print('E ' + str(np.reshape(np.dot(self.args['spikes_e'], self.args['lee']), self.args['ne'],)))
        self.args['ve'] = self.args['ve'] + dve 

    def update_vi(self):
        dvi = (self.args['dt'] / self.args['tauI']) * ((self.args['vrest'] - self.args['vi']) +
                np.reshape(np.dot(sum(self.args['mie']), np.transpose(self.args['lie'])), self.args['ni'],) -
                np.reshape(np.dot(sum(self.args['mii']), self.args['lii']), self.args['ni'],)
                + abs(np.random.randn(self.args['ni'])))
        self.args['vi'] = self.args['vi'] + dvi 

    def update_re(self):
        dre = (self.args['dt'] / self.args['tau']) * (-self.args['re'] + self.args['spikes_e'])
        self.args['re'] = self.args['re'] + dre

    def update_x(self):
        ## on E from E
        dxEE = (self.args['dt'] / self.args['tau']) * (-self.args['xee'] + self.args['tau'] * self.args['spikes_e'])
        self.args['xee'] = self.args['xee'] + dxEE
        ## on E from I 
        dxEI = (self.args['dt'] / self.args['tau']) * (-self.args['xei'] + self.args['tau'] * self.args['spikes_i'])
        self.args['xei'] = self.args['xei'] + dxEI
        ## on I from E 
        dxIE = (self.args['dt'] / self.args['tau']) * (-self.args['xie'] + self.args['tau'] * self.args['spikes_e'])
        self.args['xie'] = self.args['xie'] + dxIE
        ## on I from I
        dxII = (self.args['dt'] / self.args['tau']) * (-self.args['xii'] + self.args['tau'] * self.args['spikes_i'])
        self.args['xii'] = self.args['xii'] + dxII

    def update_m(self):
        ## on E from E
        dmEE = (self.args['dt'] / self.args['tau']) * (-self.args['mee'] + self.args['xee'])
        self.args['mee'] = self.args['mee'] + dmEE
        ## on E from I 
        dmEI = (self.args['dt'] / self.args['tau']) * (-self.args['mei'] + self.args['xei'])
        self.args['mei'] = self.args['mei'] + dmEI
        ## on I from E 
        dmIE = (self.args['dt'] / self.args['tau']) * (-self.args['mie'] + self.args['xie'])
        self.args['mie'] = self.args['mie'] + dmIE
        ## on I from I
        dmII = (self.args['dt'] / self.args['tau']) * (-self.args['mii'] + self.args['xii'])
        self.args['mii'] = self.args['mii'] + dmII

    def update_lee(self):
        dlee = np.repeat(np.dot(self.args['d'], self.args['ds']), self.args['ne']).reshape(self.args['ne'], self.args['ne'])
        dlee = dlee * 2 * self.args['eta'] * np.transpose(self.args['re'])
        if(self.args['frac_tuned'] != 1):
            self.args['lee'] = relu(self.args['lee'] - apply_mask(self.args['mask'], dlee))
        else: 
            self.args['lee'] = relu(self.args['lee'] - dlee)
        if(self.args['connectivity'] != 1):
            self.args['lee'] = apply_mask(self.args['mask_lee'], self.args['lee'])
        np.fill_diagonal(self.args['lee'], 0)

class Sim(object):
    """ A simulation of a neural network.
        args: the arguments dictionary with values set 
    """

    def __init__(self, NNt, args):
        args['NNt'] = NNt
        args['sdf'] = pd.DataFrame(np.zeros((args['steps'], args['n_stim']))) ### stim values
        args['ve_df'] = pd.DataFrame(np.zeros((args['steps'], args['ne'])))
        args['vi_df'] = pd.DataFrame(np.zeros((args['steps'], args['ni'])))
        args['spikes_e_df'] = pd.DataFrame(np.zeros((args['steps'], args['ne'])))
        args['spikes_i_df'] = pd.DataFrame(np.zeros((args['steps'], args['ni'])))
        self.args = args
        self.args['NNt'].update_ds()
        self.args['NNt'].update_s()  ## update stim value
        self.update_sdf(0)
        self.update_spikes_e(0)

    
    def update_sdf(self, t):
        """ Update remembered stimulus in dataframe sdf """
        self.args['sdf'].ix[t] = self.args['s'][0]
    
    def update_spikes_e(self, t):
        self.args['spikes_e_df'].ix[t] = self.args['NNt'].args['spikes_e']
    
    def update_spikes_i(self, t):
        self.args['spikes_i_df'].ix[t] = self.args['NNt'].args['spikes_i']

    def update_ve_df(self, t):
        self.args['ve_df'].ix[t] = self.args['NNt'].args['ve'] 
    
    def update_vi_df(self, t):
        self.args['vi_df'].ix[t] = self.args['NNt'].args['vi']
    
    def run(self):
        t = 1
        while(t < self.args['steps']):
            ### update synaptic helper variables
            self.args['NNt'].update_x()
            self.args['NNt'].update_m() 
            ### update voltages 
            self.args['NNt'].update_ve() ## update voltages_e
            self.args['NNt'].update_vi() ## update voltages_i
            #self.update_ve_df(t)
            #self.update_vi_df(t)

            ### update spikes 
            self.args['NNt'].update_spikes()
            self.update_spikes_e(t)
            self.update_spikes_i(t)

            ### update firing rates
            self.args['NNt'].update_re() ## firing rates
            
            ### update change in stim 
            self.args['NNt'].update_ds()

            ### update synaptic weights 
            if(self.args['tuned']):
                self.args['NNt'].update_lee() ## tune weights

            ### update stimulus value 
            self.args['NNt'].update_s()  ## update stim value
            self.update_sdf(t)

            ### reset voltages 
            self.args['ve'][self.args['ve'] >= self.args['vthr']] = self.args['vreset']
            self.args['vi'][self.args['vi'] >= self.args['vthr']] = self.args['vreset']
            t += 1
        return self.args         

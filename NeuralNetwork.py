""" A neural network object """
import numpy as np
import math
from scipy import linalg
from random import randint
import copy
import math 

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

def add_noise(to, size):
    """ Creates noise with specified size """
    noise = to * size * np.random.normal()
    return noise 

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
        if 'seed' in args:
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
                args['L'] = np.random.randn(args['n_neurons'], args['n_neurons'])
                args['L'] = np.divide(args['L'], math.sqrt(args['n_neurons'])) 
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
        self.args['a'] = self.args['a']*(1-self.args['dt']) + self.args['dt']*(np.dot(self.args['L'], self.args['r']))

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
            dsdt = (np.transpose(np.dot(np.transpose(self.args['d']*d_relu(self.args['a'])),  (-self.args['a'] + np.dot(self.args['L'],self.args['r'])))))[0]
        else:
            dsdt = np.vdot(self.args['d']*d_relu(self.args['a']), (-self.args['a'] + np.dot(self.args['L'],self.args['r'])))
        if(self.args['error'] == 'binary'):
            return(binary_updates(dsdt))
        return dsdt 
        
    def update_L(self):
        """ Update the connectivity matrix, L (Equation 3, 4) """
        drdt = self.drdt()
        dsdt = self.calc_dsdt()
        dL = (np.repeat(np.dot(self.args['q'],dsdt), self.args['n_neurons'])).reshape(self.args['n_neurons'],self.args['n_neurons']) 
        dL = dL * 2 * self.args['eta'] * np.transpose(self.args['r']) * drdt
        if(self.args['L_noise'] != 0):
            if(self.args['FEVER']):
                dL = self.args['L_noise'] * np.random.normal()
            elif(self.args['noise_amp'] == 'alpha'):
                dL = dL + self.args['L_noise'] * np.random.normal()
            else: 
                dL += add_noise(dL, self.args['L_noise'])
        if(self.args['frac_tuned'] != 1):
            self.args['L'] = self.args['L'] - apply_mask(self.args['mask'], dL)
        else:
            self.args['L'] = self.args['L'] - dL 
        if(self.args['connectivity'] < 1):
            self.args['L'] = apply_mask(self.args['connectivity_mask'], self.args['L'])

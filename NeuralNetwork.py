import numpy as np
import math
from scipy import linalg
from random import randint
import copy
import math 

def avg(x):
    """ Returns the average of x """
    return sum(x) / len(x)

def vect_pinv(v):
    """ Takes the Moore-Penrose pseudoinverse of a vector v"""
    vT = np.transpose(v)
    vTv = np.dot(vT, v)
    vpinv = np.true_divide(vT, vTv)
    return(vpinv)

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
    return np.maximum(0,x)

def d_relu(x):
    """ Return 1 if x >0, else 0 """
    x[x>0] = 1
    x[x<0] = 0
    return x

class NN(object):
    """ A rate-based neural network.

    Attributes:
        n_neurons: number of neurons in the network
        plastic_synapse (False): If synapses are plastic or not
        activation (relu): activation function for the firing rates of the neurons, can also run as linear 
        seed (None): optional seed for random initializations
        FEVER (False): boolean initialize with FEVER requirement d = d*L
        n_stim (None): number of stim, default 1
        a (None): activity vector (avg firing rate),  default random guassian
        d (None): weighted contribution of neurons to stim for updates, default random gaussian
        L (None): connectivity matrix, default random guassian with variance sqrt(n)
        q (None): weighted contribution of neurons to stim for stim calculation, default random gaussian
                  (same as d unless rws == True)
        frac_tuned (1): fraction of neurons to be tuned
        L_noise (0): amplitude of noise to be added to the synapse
        rws (False): use random feedback weights (q != d)
        connectivity (1): fraction of synapses to be connected 
    """
        
    def __init__(self, NN_copy=None, n_neurons=None, plastic_synapse=False, activation = 'relu',  seed = None,
                 FEVER=False, n_stim=None, a = None, d= None, L=None, q=None, frac_tuned=1, L_noise = 0, rws = False, connectivity = 1):
        """ Return a new NN object from specified parameters or copies parameters from NN_copy (but with unique a,d,L,q) """
        if(NN_copy is None):
            ###### NETWORK SETUP ######
            self.n_neurons = n_neurons
            self.plastic_synapse = plastic_synapse
            self.FEVER = FEVER
            if(FEVER):
                self.activation = 'linear'
            else:
                self.activation = activation 
            if seed is not None:
                np.random.seed(seed)
            if n_stim is None:
                if d is None:
                    self.n_stim = 1
                else:
                    self.n_stim = d.shape[1]
            else:
                self.n_stim = n_stim
            self.connectivity = connectivity
            self.frac_tuned = frac_tuned
            self.L_noise = L_noise
        else:
            self.n_neurons = NN_copy.n_neurons
            self.plastic_synapse = NN_copy.plastic_synapse
            self.FEVER = NN_copy.FEVER
            self.n_stim = NN_copy.n_stim
            self.activation = NN_copy.activation
            self.connectivity = NN_copy.connectivity
            self.rws = NN_copy.rws
            self.frac_tuned = NN_copy.frac_tuned
            self.L_noise = NN_copy.L_noise 
        ###### RANDOM INITIALIZATIONS ######
        if(a is None or NN_copy):
            self.a = np.random.randn(self.n_neurons, 1)
        else:
            self.a = a 
        if(self.activation=='relu'):
            self.r = relu(self.a)
        else:
            self.r = self.a
        if(d is None or NN_copy):
            self.d = abs(np.random.randn(self.n_neurons, self.n_stim))
        else:
            self.d = d
        self.rws = rws 
        if(rws):
            if (NN_copy or q is None):
                self.q = abs(np.random.randn(self.n_neurons, self.n_stim))
            else:
                self.q = q 
        else:
            self.q = self.d 
        if(FEVER):
            dpinv = vect_pinv(self.d)
            self.L = np.outer(self.d, dpinv)
        elif(NN_copy or L is None):
            L = np.random.randn(self.n_neurons, self.n_neurons)
            self.L = np.divide(L, math.sqrt(self.n_neurons)) 
        else:
            self.L = L
        if(self.connectivity < 1):
            self.connectivity_mask = mask_matrix(self.n_neurons, connectivity)
        if(self.frac_tuned != 1):
            self.mask = mask_matrix(self.n_neurons, frac_tuned)   
    
    def calc_s(self):
        """ Calculate and return the rememberd stim value (Equation 5) q==d unless rws==True """
        return np.dot(np.transpose(self.r), self.q)

    def update_a(self, dt):
        """ Update activity of network (Equation 1) """
        r = copy.deepcopy(self.r)
        L = copy.deepcopy(self.L)
        self.a = self.a*(1-dt) + dt*(np.dot(L,r))

    def update_r(self):
        """ Update firing rates of network """
        a = copy.deepcopy(self.a)
        if(self.activation=='relu'): 
            self.r = relu(a)
        else: ##linear 
            self.r = a

    def drdt(self):
        """ Calculate and return the derivative of the firing rates """
        a = copy.deepcopy(self.a)
        if(self.activation=='relu'):
            return d_relu(a)
        else: ##linear 
            return 1 

    def calc_dsdt(self):
        """ Calculate and return change in stim """
        a = copy.deepcopy(self.a)
        L = copy.deepcopy(self.L)
        d = copy.deepcopy(self.d)
        r = copy.deepcopy(self.r)
        drdt = self.drdt()
        if(self.n_stim >1):
            dsdt = (np.transpose(np.dot(np.transpose(d*d_relu(copy.deepcopy(a))),  (-a + np.dot(L,r)))))[0]
        else:
            dsdt = np.vdot(d*d_relu(copy.deepcopy(a)), (-a + np.dot(L,r)))
        return dsdt 
        
    def update_L(self, eta):
        """ Update the connectivity matrix, L (Equation 3, 4) """
        a = copy.deepcopy(self.a) 
        r = copy.deepcopy(self.r)
        drdt = self.drdt()
        d = copy.deepcopy(self.d)
        dsdt = self.calc_dsdt()
        dL = (np.repeat(np.dot(d,dsdt), self.n_neurons)).reshape(self.n_neurons,self.n_neurons) 
        dL = dL * 2 * eta * np.transpose(r) * drdt
        if(self.L_noise != 0):
            dL += add_noise(dL, self.L_noise)
        if(self.frac_tuned != 1):
            self.L = self.L - apply_mask(self.mask, dL)
        else:
            self.L = self.L - dL 
        if(self.connectivity < 1):
            self.L = apply_mask(self.connectivity_mask, self.L)

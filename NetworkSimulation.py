from NeuralNetwork import NN
import pandas as pd
import numpy as np
from scipy import linalg

def avg(x):
    return sum(x) / len(x)

class Sim(object):
    """ A simulation of a neural network.
    Attributes:
        NN: The neural network object to run 
        steps (100): number of steps to run
        ms (None) number of ms to run the iteration for, overwrites steps
        updates (True): If remembered stimulus updates should be printed every 100 steps
        eta (None): the learning rate, default for 100 and 1000 neurons set
        dt (None): the time scale, default for 100 and 1000 neurons set
        tau (1): approximately 20 ms
        n_activities (10): number of neurons to keep track of activities for
        sdf: The dataframe that holds remembered stim over time
        dsdts: the dsdt values over time
        update_frs: Store firing rates to recreate Figure 2A
    """

    def __init__(self, Sim_copy = None, NNet=None, steps=100, ms=None, updates=False, dt = None,
                 eta = None, tau=1, n_rs = 10, update_rs=False):
        """ Returns a new NetworkSimulation object with specified parameters or copies parameters from Sim_copy """
        if(Sim_copy is not None):
            self.NNet = NN(NN_copy = Sim_copy.NNet)
            self.ms = Sim_copy.ms
            self.updates = Sim_copy.updates
            self.dt = Sim_copy.dt
            self.eta = Sim_copy.eta
            self.tau = Sim_copy.tau
            self.update_rs = Sim_copy.update_rs
            self.n_rs = Sim_copy.n_rs
            self.steps = Sim_copy.steps 
        else:
            ######## calculate steps to run
            self.NNet = NNet
            if dt is None:
                self.dt = .001
            else:
                self.dt = dt
            if ms is not None:
                self.steps = ms / (self.dt * tau * 20)
                self.ms = ms 
            elif steps is not None:
                self.steps = steps
                self.ms = self.steps*self.dt*tau*20
            self.updates = updates
            if eta is None:
                self.eta = .0001
            else:
                self.eta = eta
            self.tau = tau
            self.update_rs = update_rs
            self.n_rs = n_rs 

        self.stepsperms = 1 / (self.dt*tau*20)
        ######## storing info 
        if(self.update_rs):
            self.frs = pd.DataFrame(np.zeros((self.ms,self.n_rs)))
            if(self.n_rs < self.NNet.n_neurons):
                self.rcols = np.random.choice(list(range(self.NNet.n_neurons)), self.n_rs)
        self.initial_stim = self.NNet.calc_s()
        self.sdf = pd.DataFrame(np.zeros((self.ms, self.NNet.n_stim))) ### stim vlaues 
        self.phis = pd.DataFrame(np.zeros((self.ms, self.NNet.n_stim)))
        if(self.NNet.frac_tuned ==0):
            self.NNet.plastic_synapse=False
        ######## PRINT NETWORK SETUP 
        if(self.NNet.plastic_synapse):
            frac = self.NNet.frac_tuned * 100
            setup = str(frac) +'%' +  ' Tuned'
        elif(self.NNet.FEVER==True):
            setup = 'FEVER'
        else:
            setup='Untuned'
        print(setup + ' '+ str(self.NNet.activation) + ' activation; ' +str(self.NNet.n_neurons) +
              ' neurons; ' + str(self.steps) + ' steps; dt=' +str(self.dt)+'; eta=' + str(self.eta))
    
    def update_sdf(self, t):
        """ Update remembered stimulus in dataframe sdf """
        s = self.NNet.calc_s()
        if(t == 0):
            mst = 0
        else:
            mst = int(t*self.dt*self.tau*20)
        for i in range(self.NNet.n_stim):
            self.sdf[i][mst] = s[0][i]

    def update_frs(self, t):
        """ Update the neural activities in dataframe activities """
        if(t == 0):
            mst = 0
        else:
            mst = int(t*self.dt*self.tau*20)
        if(len(self.rcols) == self.NNet.n_neurons):
            self.frs.ix[mst] = self.NNet.r 
        else:
            frs = [r for sublist in self.NNet.r[self.rcols].tolist() for r in sublist]
            self.frs.ix[mst] = frs

    def calc_phi(self):
        """Returns scaled change in stim: 1/s(t_final) * dsdt_final """
        phis = abs(self.NNet.calc_dsdt() / self.NNet.calc_s())
        return phis

    def update_phis(self, t):
        """ Update phis (scaled change in stimulus)"""
        if(t == 0):
            mst = 0
        else:
            mst = int(t*self.dt*self.tau*20)
        p = self.calc_phi()
        for i in range(self.NNet.n_stim):
            self.phis[i][mst] = p[0][i]
        
    def run(self):
        """ Run a network simulation """
        t = 0 
        while(t < self.steps):
            if(t % self.stepsperms == 0):
                self.update_sdf(t) ## storing stim values 
                if(self.update_rs): ## storing firing rates
                    self.update_frs(t)
                if(self.updates): ## printing updates 
                    print(str(int(t*self.dt*self.tau*20)) + ' ms')
                    print(self.NNet.calc_s())
                self.update_phis(t) ## storing phi values
            self.NNet.update_a(self.dt)
            self.NNet.update_r()
            if(self.NNet.plastic_synapse):
                self.NNet.update_L(self.eta)
            t += 1
        
            
            
            
            
            
            
            

""" a network simulation object """

from NeuralNetwork import NN
import pandas as pd
import numpy as np
from scipy import linalg

def avg(x):
    return sum(x) / len(x)

class Sim(object):
    """ A simulation of a neural network.
        args: the arguments dictionary with values set 
    """

    def __init__(self, NNt, args):
        """ Returns a new NetworkSimulation object with specified parameters or copies parameters from Sim_copy """
        args['NNt'] = NNt
        args['steps'] = args['ms'] / (args['dt'] * args['tau'] * 20)
        args['stepsperms'] = 1 / (args['dt']*args['tau']*20)
        ######## storing info 
        if(args['store_frs']):
            args['frs'] = pd.DataFrame(np.zeros((args['ms'], args['nrs'])))
            args['frs_idx'] = np.random.choice(list(range(args['n_neurons'])), args['nrs'])
        args['initial_stim'] = args['NNt'].calc_s()
        args['sdf'] = pd.DataFrame(np.zeros((args['ms'], args['n_stim']))) ### stim vlaues
        if(args['frac_tuned'] == 0):
            args['tuned'] = False
        self.args = args
    
    def update_sdf(self, t):
        """ Update remembered stimulus in dataframe sdf """
        s = self.args['NNt'].calc_s()
        if(t == 0):
            mst = 0
        else:
            mst = int(t*self.args['dt']*self.args['tau']*20)
        for i in range(self.args['n_stim']):
            self.args['sdf'][i][mst] = s[0][i]

    def update_frs(self, t):
        """ Update the neural activities in dataframe activities """
        if(t == 0):
            mst = 0
        else:
            mst = int(t*self.args['dt']*self.args['tau']*20)
        frs = [r for sublist in self.args['r'][self.args['frs_idx']].tolist() for r in sublist]
        self.args['frs'].ix[mst] = frs

    def run(self):
        """ Run a network simulation """
        t = 0 
        while(t < self.args['steps']):
            if(t % self.args['stepsperms'] == 0):
                self.update_sdf(t) ## storing stim values 
                if(self.args['store_frs']): ## storing firing rates
                    self.update_frs(t)
                if(self.args['updates']): ## printing updates 
                    print(str(int(t*self.args['dt']*self.args['tau']*20)) + ' ms')
                    print(self.args['NNt'].calc_s())
            self.args['NNt'].update_a()
            self.args['NNt'].update_r()

            if(self.args['tuned']):
                self.args['NNt'].update_L()
            t += 1
        return self.args 
            

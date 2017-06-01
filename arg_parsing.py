import argparse
import numpy as np 

class my_args(object):
    def __init__(self, args):
        if args.milliseconds:
            self.ms = args.milliseconds
        else:
            self.ms = 3001
        if args.num_neurons:
            self.n_neurons = args.num_neurons
        else:
            self.n_neurons = 100
        if args.time_scale:
            self.dt = args.time_scale
        else:
            self.dt = .001
        if args.learning_rate:
            self.eta = args.learning_rate
        else:
            self.eta = .0001
        if args.activation:
            self.activation = args.activation
        else:
            self.activation = 'relu'
        if args.num_stim:
            self.n_stim = args.num_stim
        else:
            self.n_stim = 1
        if args.num_firing_rates:
            self.nrs = args.num_firing_rates
        else:
            self.nrs = 10
        if args.seed:
            self.seed = 19
        else:
            self.seed = np.random.randint(0,1000)
        if args.num_iterations:
            self.n_iters = args.num_iterations
        else:
            self.n_iters = 100 

    def ms(self):
        return self.ms
    
    def n_neurons(self):
        return self.n_neurons
    
    def dt(self):
        return self.dt
    
    def eta(self):
        return self.eta
    
    def activation(self):
        return self.activation
    
    def n_stim(self):
        return self.n_stim
    
    def nrs(self):
        return self.nrs
    
    def seed(self):
        return self.sd
    
def get_args():
    parser = argparse.ArgumentParser(
            description="Generic parser to get arguments from command line"
        )
    parser.add_argument('-ms', '--milliseconds', action='store',
                        type=int,
                        required=False,
                        help='Number of milliseconds')
    parser.add_argument('-n', '--num_neurons', action='store',
                        type=int,
                        required=False,
                        help='Number of neurons')
    parser.add_argument('-dt', '--time_scale', action='store',
                        type=float,
                        required=False,
                        help='The time scale')
    parser.add_argument('-eta', '--learning_rate', action='store',
                        type=float,
                        required=False,
                        help='The learning rate')
    parser.add_argument('-a', '--activation', action='store',
                        type = str,
                        choices=['relu', 'linear'], 
                        required=False,
                        help='Activation function')
    parser.add_argument('-s', '--num_stim', action='store',
                        type = int, 
                        required=False,
                        help='Number of stim')
    parser.add_argument('-f', '--num_firing_rates', action='store',
                        type = int, 
                        required=False,
                        help='Number of firing rates to plot')
    parser.add_argument('-sd', '--seed', action='store',
                        type=bool, 
                        choices=[True, False], 
                        required=False,
                        help='Use the same seed in the paper or not')
    parser.add_argument('-i', '--num_iterations', action='store',
                        type=int, 
                        required=False,
                        help='The number of iterations to run')
    args = parser.parse_args()
    args_obj = my_args(args)
    return args_obj 



    

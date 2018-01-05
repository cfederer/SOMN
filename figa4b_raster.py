import matplotlib.pyplot as plt
from NN_LIF import *
from plot_util import *

ms = 50
seed = np.random.randint(10000)
print('seed : ' + str(seed))

args = get_args()
args['ne'] = 100
args['ms'] = ms
args['seed'] = seed 
args['tuned'] = True
nn_lif = NN(args)

sim_lif = Sim(nn_lif, args)
sim_lif.run()

argsu = get_args()
argsu['ms'] = ms
argsu['seed'] = seed
argsu['tuned'] = False
nn_lif_u = NN(argsu)
sim_lifu = Sim(nn_lif_u, argsu)
sim_lifu.run()


pa = plot_args()
plot_raster(pa, args)

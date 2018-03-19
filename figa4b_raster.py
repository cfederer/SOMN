import matplotlib.pyplot as plt
from NN_LIF import *
from plot_util import *

ms = 10
seed = 0 
#seed = np.random.randint(10000)
print('seed : ' + str(seed))

args = get_args()
args['ne'] = 100
args['ms'] = ms
args['seed'] = seed 
args['tuned'] = True
args['gee'] = 2
#args['gei'] = 100000
nn_lif = NN(args)

sim_lif = Sim(nn_lif, args)
sim_lif.run()
'''
argsu = get_args()
argsu['ms'] = ms
argsu['seed'] = seed
argsu['tuned'] = False
nn_lif_u = NN(argsu)
sim_lifu = Sim(nn_lif_u, argsu)
sim_lifu.run()
'''
plt.plot(args['sdf'])
plt.show()
plt.close()
pa = plot_args()
plot_raster(pa, args)

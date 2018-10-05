"""Recreates Figure 2A from bioRxiv: 144683 (Federer & Zylberbeg 2017)"""
from NN import * 
import pandas as pd
from plot_util import *
from numpy import diff

ms = 3000

## run plastic random synapse network 
args_t = get_args()
args_t['seed'] = 0 
args_t['tuned'] = True
args_t['g'] = 1.6
args_t['store_frs'] = True
args_t['ms'] = ms

NNt = NN(args_t)
Simt=Sim(NNt, args_t)
Simt.run()

fig = plt.gcf()
differentials = list()
for i in range(args_t['nrs']):
    dy = diff(args_t['frs'][i])
    differentials.append(dy)
    plt.plot(dy)


ax = fig.add_subplot(111)
ax.set_title('dr/dt of 10 Neurons with g = ' + str(args_t['g']))
ax.set_xlabel('Time (ms)')
ax.set_ylabel('dr/dt')
fig.savefig('plots/' + 'dr_dt_g=' + str(args_t['g']) + '.eps', dpi=1200)
#fig.savefig('plots/dr_dt_g=' + str(args_t['g']) + '.png')

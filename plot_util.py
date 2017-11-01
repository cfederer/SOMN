"""helper code for plotting """

import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.patches import Rectangle
from scipy import stats


def plot_args(poster=False):
    """ default arguments for plotting """
    pa = dict()
    pa['poster'] = poster   ### larger fonts 
    if(poster):
        pa['text_size'] = 20
        pa['label_size'] = 23
        pa['tick_label_size'] = 18
        pa['linewidth'] = 3.5
    else: 
        pa['text_size'] = 10
        pa['label_size'] = 15 
        pa['tick_label_size'] = 10
        pa['linewidth'] = 4.0
    pa['xlabel'] = 'Time (ms)'
    pa['ylabel'] = 'Frac Stim Retained'
    pa['colors'] = ['DarkRed', 'red','orangered', 'darkorange',
            'limegreen', 'teal', 'DodgerBlue',
            'blue','darkblue', 'purple', 'black']
    pa['error_colors'] = ['#ff3333', '#ff8080','#ffb499', '#ffd199',
             '#c1f0c1', '#ccffff', '#b3d9ff', 
             '#8080ff', '#6666ff','#ccb3e6', 'gray']
    pa['plt_errors'] = False
    pa['save'] = False
    pa['save_as'] = str(abs(np.round(np.random.randn(), 4)))
    pa['show'] = True
    pa['right_margins'] = False 
    return pa

def plot_pqs(dfs, pa):
    """ plots frac of stim remained results from quantify_sims_multithreading"""
    means = list()
    errors = list()
    for i in range(len(dfs)):
        means.append(dfs[i].mean(axis=1).as_matrix())
        if(pa['plt_errors']):
            errors.append(stats.sem(dfs[i].as_matrix().transpose()))
    #fig = plt.figure()
    fig = plt.gcf()
    if(pa['right_margins']):
            fig.subplots_adjust(right=.75)
    
    ax = fig.add_subplot(111)
    ax.set_title(' ', fontweight='bold')
    ax.set_xlabel(pa['xlabel'], fontsize=pa['label_size'] , fontweight='bold')
    ax.set_ylabel(pa['ylabel'], fontsize=pa['label_size'], fontweight='bold')
    x = list(range(len(dfs[0])))
    if('xspot' in pa):
        xspot = pa['xspot']
    else:
        xspot = 0
    pylab.xlim([0, len(x)-1])
    if('ylim' in pa):
        pylab.ylim(pa['ylim'])
    if('yadds' not in pa):
        pa['yadds'] = np.repeat(0, len(dfs))
    if('linestyle' not in pa):
        pa['linestyle'] = np.repeat('-', len(dfs))
    elif(type(pa['yadds']) is not list):
        pa['yadds'] = np.repeat(pa['yadds'], len(dfs))
    for i, df in enumerate(dfs):
        t, = plt.plot(x, means[i], pa['colors'][i], label = pa['labels'][i], linestyle = pa['linestyle'][i], linewidth=pa['linewidth'])
        if(pa['plt_errors']):
            plt.fill_between(x, means[i] - errors[i], means[i] + errors[i], color=pa['error_colors'][i])
        if(xspot < len(x)):
            yspot = means[i][xspot]
        else:
            yspot = means[i][len(x)-1]
        yspot += + pa['yadds'][i]
        plt.text(xspot, yspot, pa['labels'][i], color=pa['colors'][i], fontsize=pa['text_size'], fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])
    if(not pa['right_margins']):
         plt.tight_layout()
    if(pa['show']):  
        plt.show()
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.pdf')
    
    plt.close()
    return True
    
def plot_stims(args, pa):
    """ plots the remembered stimulus values """
    means = list()
    errors = list()
    
    for i in range(len(args)):
        means.append(args[i]['sdf'].mean(axis=1).as_matrix())
        if(pa['plt_errors']):
            errors.append(stats.sem(args[i]['sdf'].as_matrix().transpose()))
    fig = plt.gcf()
    if(pa['right_margins']):
        fig.subplots_adjust(right=.75)
    ax = fig.add_subplot(111)
    ax.set_title(' ', fontweight='bold')
    ax.set_xlabel(pa['xlabel'], fontsize=pa['text_size'] , fontweight='bold')
    ax.set_ylabel(pa['ylabel'], fontsize=pa['text_size'], fontweight='bold')
    x = list(range(len(args[0]['sdf'][0])))
    if('xspot' in pa):
        xspot = pa['xspot']
    else:
        xspot = len(x) 
    pylab.xlim([0, len(x)-1])
    if('ylim' in pa):
       pylab.ylim(pa['ylim'])
    if('yadds' not in pa):
        pa['yadds'] = np.repeat(0, len(args))
    elif(type(pa['yadds']) is not list):
        pa['yadds'] = np.repeat(pa['yadds'], len(args))
    for i in range(len(args)):
        t, = plt.plot(x, means[i], pa['colors'][i], label = pa['labels'][i], linewidth=pa['linewidth'])
        if(pa['plt_errors']):
            plt.fill_between(x, means[i] - errors[i], means[i] + errors[i], color=pa['colors'][i])
        yspot = means[i][len(x)-1] + pa['yadds'][i]
        plt.text(xspot, yspot, pa['labels'][i], color=pa['colors'][i], fontsize=pa['text_size'], fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])
    plt.show()
    plt.close()
    return True

def plot_activities(args_tuned, args_untuned, pa):
    """ Plot dynamics  for Fig. 2A """
    height = 1
    txtcolor = 'black'
    blue = 'blue'
    red = 'red'
    x = list(range(len(args_tuned['frs'][:51])))
    fig = plt.figure(1)
    fig.text(0.06, 0.5, 'Firing Rates $r_i(t)$' , ha='center', va='center',
             rotation='vertical', fontsize=pa['text_size'], color=txtcolor, fontweight='bold')

    ####### Constant Synapse section 
    ax2 = fig.add_subplot(212)
    t1 =ax2.set_title('Constant Synapse', fontsize =pa['text_size'], fontweight='bold', color='black')
    t1.set_position([.5, 1.12])

    pylab.ylim([0, height])
    pylab.xlim([0, len(args_untuned['frs'][:51])-1])
    xspots = [0, 17, 35]
    txspots = [xspots[0] , xspots[1], xspots[2]] 
    tyspot = height + .01
    yspot = 0
    currentAxis = plt.gca()
    ax2.set_xlabel('Time (ms) ', fontsize=pa['text_size'], color=txtcolor, fontweight='bold')
    for i in range(len(xspots)):
        currentAxis.add_patch(Rectangle((xspots[i], -.5), height, 3, facecolor="lightgrey", edgecolor=blue))### add gray bars
        plt.text(txspots[i], tyspot, r'$s = \sum_{i=0} d_i r_i$' +'=' + str(np.round(args_untuned['sdf'][0][int(xspots[i])], 2)), color=blue, fontsize =pa['text_size'], fontweight='bold') ### add text 
    for i in range(len(args_untuned['frs'].columns)):
        a, = plt.plot(x, args_untuned['frs'][i][:51],  red, linestyle='--',linewidth=2.0)

    ###### Plastic synapse section 
    ax1 = fig.add_subplot(211)
    t2 = ax1.set_title('Plastic Synapse', fontsize = pa['text_size'], fontweight='bold', color='black')
    t2.set_position([.5, 1.14])
    ax1.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
    pylab.ylim([0, height])
    pylab.xlim([0, len(args_untuned['frs'][:51])-1])
    currentAxis = plt.gca()
    for i in range(len(xspots)):
        currentAxis.add_patch(Rectangle((xspots[i], -.5), height, 3, facecolor="lightgrey", edgecolor=blue)) ### add gray bars
        plt.text(txspots[i], tyspot, r'$s = \sum_{i=0} d_i r_i$' +'=' + str(np.round(args_tuned['sdf'][0][int(xspots[i])], 2)), color=blue,fontsize =pa['text_size'], fontweight='bold') ### add text 
    for i in range(len(args_tuned['frs'].columns)):
        a, = plt.plot(x,args_tuned['frs'][i][:51], red, linestyle='--', linewidth=2.0)

    ## plot final 
    plt.subplots_adjust(hspace = .3)
    plt.setp(ax1.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax1.get_yticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax2.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax2.get_yticklabels(), fontsize=pa['tick_label_size'])
    if(pa['show']):
        plt.show()
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.pdf')
    plt.close() 
    return True

from NeuralNetwork import NN
from NetworkSimulation import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import pylab
from matplotlib import rc
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from scipy import stats
mpl.rc('font', family='Arial')
ylabel_size = 20
xlabel_size = 20
txtlabel_size = 20
tick_label_size = 15
std_xlabel = 'Time (ms)'
std_ylabel = 'Fraction of Stim Retained'
linewidth = 3.0

d_colors = ['DarkRed', 'red','orangered', 'darkorange',
            'limegreen', 'teal', 'DodgerBlue',
            'blue','darkblue', 'purple']

da_colors = ['#ff3333', '#ff8080','#ffb499', '#ffd199',
             '#c1f0c1', '#ccffff', '#b3d9ff', 
             '#8080ff', '#6666ff','#ccb3e6']

def plot_activities(stimsu, actsu, stimst,  actst, dt=.01, tau=1):
    txtlabel_size = 20
    title_size = 20
    """ Plot dynamics  """
    height = 1
    txtcolor = 'black'
    blue = 'blue'
    red = 'red'
    x = list(range(len(actsu)))
    fig = plt.figure(1)
    fig.text(0.06, 0.5, 'Firing Rates $r_i$' , ha='center', va='center', rotation='vertical', fontsize=ylabel_size, color=txtcolor, fontweight='bold')
    ####### Constant Synapse section 
    ax1 = fig.add_subplot(211)
    t1 =ax1.set_title('Constant Synapse', fontsize =title_size, fontweight='bold', color='black')
    t1.set_position([.5, 1.15])
    ax1.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
    pylab.ylim([0, height])
    pylab.xlim([0, len(actsu)-1])
    xspots = [0, 17, 35]
    txspots = [xspots[0] , xspots[1], xspots[2]] 
    tyspot = height + .01
    yspot = 0
    currentAxis = plt.gca()
    for i in range(len(xspots)):
        currentAxis.add_patch(Rectangle((xspots[i], -.5), height, 3, facecolor="lightgrey", edgecolor=blue))### add gray bars
        plt.text(txspots[i], tyspot, r'$s = \sum_{i=0} d_i r_i$' +'=' + str(np.round(stimsu[0][int(xspots[i])], 2)), color=blue, fontsize =txtlabel_size, fontweight='bold') ### add text 
    for i in range(len(actsu.columns)):
        a, = plt.plot(x, actsu[i],  red, linestyle='--',linewidth=2.0)
    ax2 = fig.add_subplot(212)
    t2 = ax2.set_title('Plastic Synapse', fontsize = title_size, fontweight='bold', color='black')
    t2.set_position([.5, 1.15])
    ax2.set_xlabel('Time (ms) ', fontsize=xlabel_size, color=txtcolor, fontweight='bold')
    pylab.ylim([0, height])
    pylab.xlim([0, len(actsu)-1])
    currentAxis = plt.gca()
    for i in range(len(xspots)):
        currentAxis.add_patch(Rectangle((xspots[i], -.5), height, 3, facecolor="lightgrey", edgecolor=blue)) ### add gray bars
        plt.text(txspots[i], tyspot, r'$s = \sum_{i=0} d_i r_i$' +'=' + str(np.round(stimst[0][int(xspots[i])], 2)), color=blue,fontsize =txtlabel_size, fontweight='bold') ### add text 
    for i in range(len(actst.columns)):
        a, = plt.plot(x,actst[i], red, linestyle='--', linewidth=2.0)
    plt.subplots_adjust(hspace = .3)
    plt.setp(ax1.get_xticklabels(), fontsize=tick_label_size)
    plt.setp(ax1.get_yticklabels(), fontsize=tick_label_size)
    plt.setp(ax2.get_xticklabels(), fontsize=tick_label_size)
    plt.setp(ax2.get_yticklabels(), fontsize=tick_label_size)
    plt.show()
    plt.close() 
    return True

def plot_stims(dfs, labels, colors=None, a_colors = None, dt=.01, xlabel=None,
               ylabel = None, plt_errors=True, yadds=None, 
               label_size=None, ylim = None, xspot = None):
    if colors is None:
        colors = d_colors
    if a_colors is None:
        a_colors = da_colors
    if xlabel is None:
        xlabel = std_xlabel
    if ylabel is None:
        ylabel = std_ylabel
    if label_size is None:
        label_size = txtlabel_size
    means = list()
    if(plt_errors):
        errors = list()
    for i in range(len(dfs)):
        means.append(dfs[i].mean(axis=1).as_matrix())
        if(plt_errors):
            errors.append(stats.sem(dfs[i].as_matrix().transpose()))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(' ', fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=xlabel_size, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight='bold')
    x = list(range(len(dfs[0])))
    if(xspot is None):
        xspot = len(x) 
    else:
        xspot = xspot
    pylab.xlim([0, len(x)-1])
    if(ylim is not None):
       pylab.ylim(ylim)
    if(yadds is None):
        yadds = np.repeat(0, len(dfs))
    elif(type(yadds) is not list):
        yadds = np.repeat(yadds, len(dfs))
    for i in range(len(dfs)):
        t, = plt.plot(x, means[i], colors[i], label = labels[i], linewidth=linewidth)
        if(plt_errors):
            plt.fill_between(x, means[i] - errors[i], means[i] + errors[i], color=a_colors[i])
        yspot = means[i][len(x)-1] + yadds[i]
        plt.text(xspot, yspot, labels[i], color=colors[i], fontsize=label_size, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=tick_label_size)
    plt.setp(ax.get_yticklabels(), fontsize=tick_label_size)
    plt.show()
    plt.close()
    return True

def plot_multidim(tdfs, udfs, tlabels, ulabels, errors=True, colors=None, a_colors=None,dt=.01,
                 xlabel=None, ylabel = None, ylim=None, xspot = None, yadds=None):

    if colors is None:
        colors = ['blue', 'orange', 'red', 'MediumSeaGreen']
    if a_colors is None:
        a_colors = ['#8080ff', '#ffdb99', '#ff9999', '#c6ecd7']
    if xlabel is None:
        xlabel = std_xlabel
    if ylabel is None:
        ylabel = std_ylabel
    tmeans = list()
    umeans = list()
    terrors = list()
    uerrors = list()
    for i in range(len(tdfs)):
        tmeans.append(tdfs[i].mean(axis=1).as_matrix())
        terrors.append(stats.sem(tdfs[i].as_matrix().transpose()))
        umeans.append(udfs[i].mean(axis=1).as_matrix())
        uerrors.append(stats.sem(udfs[i].as_matrix().transpose()))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(' ')
    ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=xlabel_size, fontweight='bold')
    x = list(range(len(tdfs[0])))
    
    if (xspot is None):
        xspot = len(x) 
    else:
        xspot = xspot
    pylab.xlim([0, len(x)-1])
    if(yadds is None):
        yadds = np.repeat(0, len(tdfs))
    elif(type(yadds) is not list):
        yadds = np.repeat(yadds, len(tdfs))
    if(ylim is not None):
        pylab.ylim(ylim)
    for i in range(len(tdfs)):
        color = colors[i]
        a, = plt.plot(x, tmeans[i], color, linewidth=2.0)
        b, = plt.plot(x, umeans[i], linestyle = '--', color=color, linewidth=2.0)
        if(errors):
            plt.fill_between(x, tmeans[i] - terrors[i], tmeans[i] + terrors[i], color=a_colors[i])
            plt.fill_between(x, umeans[i] - uerrors[i], umeans[i] + uerrors[i], color=a_colors[i])
        yspot_t = tmeans[i][len(x)-1] + yadds[i]
        yspot_u = umeans[i][len(x)-1] + yadds[i]
        plt.text(xspot, yspot_t, tlabels[i], color=color, fontsize=txtlabel_size, fontweight='bold')
        plt.text(xspot, yspot_u, ulabels[i], color=color, fontsize=txtlabel_size, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=tick_label_size)
    plt.setp(ax.get_yticklabels(), fontsize=tick_label_size)
    plt.show()
    plt.close()
    return True    


"""helper code for plotting """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pylab
from matplotlib.patches import Rectangle
from scipy import stats
from mimic_alpha import * 


def plot_args(poster=False, n_curves = 2, FEVER=False, rb=True):
    """ default arguments for plotting """
    pa = dict()
    pa['poster'] = poster   ### larger fonts 
    if(poster):
        pa['text_size'] = 20
        pa['label_size'] = 23
        pa['tick_label_size'] = 18
        pa['linewidth'] = 3.5
    else: 
        pa['text_size'] = 20
        pa['label_size'] = 15
        pa['ylabel_size'] = 22
        pa['tick_label_size'] = 12
        pa['linewidth'] = 4.0
    pa['title_size'] = pa['label_size']
    pa['title'] = ' ' 
    pa['xlabel'] = 'Time (ms)'
    #pa['ylabel'] = "Remembered Value Relative to Initial" #$r'$\frac{\hat{s}(t)}{\hat{s}(t=0)}$'"
    pa['ylabel'] = r'$\frac{\hat{s}(t)}{\hat{s}(t=0)}$'
    #bwr = plt.get_cmap('coolwarm') ##other otpions are coolwarm and bwr
    if(rb):
        bwr =  colors.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    else:
        bwr =  colors.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])
    values = range(n_curves)
    cNorm = colors.Normalize(vmin=0, vmax = values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=bwr)
    #pa['colors'] = ['DarkRed', 'red','orangered', 'darkorange',
    #       'limegreen', 'teal', 'DodgerBlue',
    #        'blue','darkblue', 'purple', 'black']
    #pa['error_colors'] = ['#ff3333', '#ff8080','#ffb499', '#ffd199',
    #         '#c1f0c1', '#ccffff', '#b3d9ff', 
    #         '#8080ff', '#6666ff','#ccb3e6', 'gray']
    pa['colors'] = list()
    for idx in range(n_curves):
        pa['colors'].append(scalarMap.to_rgba(values[idx]))
    if(FEVER):
        pa['colors'].append('k')
    pa['plt_errors'] = False
    pa['save'] = False
    pa['save_as'] = str(abs(np.round(np.random.randn(), 4)))
    pa['show'] = True
    pa['right_margins'] = False
    pa['frac_stim'] = False 
    return pa

def ordered_labels(dfs, labels):
    ordered_labels = list()
    means = list()
    last_vals = list()
    x = list(range(len(dfs[0])))
    for i in range(len(dfs)):
        mean = dfs[i].mean(axis=1).as_matrix()
        last_vals.append(mean[len(x)-1])
    while(len(last_vals)>0):
        min_idx = np.argmin(last_vals)
        ordered_labels.append(labels[min_idx])
        last_vals.pop(min_idx)
        labels.pop(min_idx)
    return ordered_labels
        

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
    ax.set_title(pa['title'], fontsize=pa['title_size'], fontweight='bold')
    ax.set_xlabel(pa['xlabel'], fontsize=pa['label_size'] , fontweight='bold')
    ax.set_ylabel(pa['ylabel'], fontsize=pa['ylabel_size'], fontweight='bold')
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
    if('text_colors' not in pa):
        pa['text_colors'] = pa['colors']
    if('linestyle' not in pa):
        pa['linestyle'] = np.repeat('-', len(dfs))
    elif(type(pa['yadds']) is not list):
        pa['yadds'] = np.repeat(pa['yadds'], len(dfs))
    for i, df in enumerate(dfs):
        t, = plt.plot(x, means[i], color=pa['colors'][i], label = pa['labels'][i], linestyle = pa['linestyle'][i], linewidth=pa['linewidth'])
        if(pa['plt_errors']):
            plt.fill_between(x, means[i] - errors[i], means[i] + errors[i], color=colorAlpha_to_rgb(pa['colors'][i], alpha=.5))
        if 'xspots' in pa:
            xspot = pa['xspots'][i]
        if 'yspots' in pa :
            yspot = pa['yspots'][i]
        else:
            if(xspot < len(x)):
                yspot = means[i][xspot]
            else:
                yspot = means[i][len(x)-1]
            yspot += + pa['yadds'][i]
        plt.text(xspot, yspot, pa['labels'][i], color=pa['text_colors'][i], fontsize=pa['text_size'], fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])
    if(not pa['right_margins']):
         plt.tight_layout()
    if(pa['show']):  
        plt.show()
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.eps', dpi=1200 )
    
    plt.close()
    return True

def plot_dfs(dfs, pa):
    """ plots the remembered stimulus values """
    print('Plotting from read in files - expects string column number')
    fig = plt.gcf()
    if(pa['right_margins']):
        fig.subplots_adjust(right=.75)
    ax = fig.add_subplot(111)
    ax.set_title(' ', fontweight='bold')
    ax.set_xlabel(pa['xlabel'], fontsize=pa['label_size'] , fontweight='bold')
    ax.set_ylabel(pa['ylabel'], fontsize=pa['ylabel_size'], fontweight='bold')
    x = list(range(len(dfs[0])))
    if('xspot' in pa):
        xspot = pa['xspot']
    else:
        xspot = len(x) 
    pylab.xlim([0, len(x)-1])
    if('ylim' in pa):
       pylab.ylim(pa['ylim'])
    if('yadds' not in pa):
        pa['yadds'] = np.repeat(0, len(dfs))
    elif(type(pa['yadds']) is not list):
        pa['yadds'] = np.repeat(pa['yadds'], len(dfs))
    for i in range(len(dfs)):
        if(pa['frac_stim']):
            stim_f = (dfs[i]['0'] / float(dfs[i]['0'][0])).values.ravel()
            t, = plt.plot(x, stim_f, color=pa['colors'][i], label = pa['labels'][i], linewidth=pa['linewidth'])
        else:
            t, = plt.plot(x, dfs[i]['0'], color=pa['colors'][i], label = pa['labels'][i], linewidth=pa['linewidth'])
        yspot = (dfs[i].loc[len(x)-1]/dfs[i].loc[0]) + pa['yadds'][i]
        #yspot = (dfs[i]['0'][len(x)-1]/dfs[0]['0'][0]) + pa['yadds'][i]
        plt.text(xspot, yspot, pa['labels'][i], color=pa['colors'][i], fontsize=pa['text_size'], fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.eps', dpi=1200 )
    if(pa['show']):
        plt.show()
    plt.close()
    return True

  
def plot_argslist(args_list, pa):
    """ plots the remembered stimulus values """
    fig = plt.gcf()
    if(pa['right_margins']):
        fig.subplots_adjust(right=.75)
    ax = fig.add_subplot(111)
    ax.set_title(' ', fontweight='bold')
    ax.set_xlabel(pa['xlabel'], fontsize=pa['label_size'] , fontweight='bold')
    ax.set_ylabel(pa['ylabel'], fontsize=pa['ylabel_size'], fontweight='bold')
    x = list(range(len(args_list[0]['sdf'][0])))
    if('xspot' in pa):
        xspot = pa['xspot']
    else:
        xspot = len(x) 
    pylab.xlim([0, len(x)-1])
    if('ylim' in pa):
       pylab.ylim(pa['ylim'])
    if('yadds' not in pa):
        pa['yadds'] = np.repeat(0, len(args_list))
    elif(type(pa['yadds']) is not list):
        pa['yadds'] = np.repeat(pa['yadds'], len(args))
    for i in range(len(args_list)):
        if(pa['frac_stim']):
            stim_f = (args_list[i]['sdf'] / float(args_list[i]['initial_stim'])).values.ravel()
            t, = plt.plot(x, stim_f, color=pa['colors'][i], label = pa['labels'][i], linewidth=pa['linewidth'])
        else:
            t, = plt.plot(x, args_list[i]['sdf'], color=pa['colors'][i], label = pa['labels'][i], linewidth=pa['linewidth'])
        yspot = (args_list[i]['sdf'][0][len(x)-1]/args_list[i]['initial_stim']) + pa['yadds'][i]
        plt.text(xspot, yspot, pa['labels'][i], color=pa['colors'][i], fontsize=pa['text_size'], fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])
    plt.show()
    plt.close()
    return True

def plot_activities(dft, dfu, frst, frsu, pa):
    """ Plot dynamics  for Fig. 2A """
    height = 2
    width = 10
    txtcolor = 'black'
    blue = 'blue'
    red = 'red'
    ms = 500
    xspots = np.linspace(0, ms-10, 3)
    txspots = [xspots[0], xspots[1]-60, xspots[2]-85]
    x = list(range(len(frst[:ms])))
    fig = plt.figure(1)
    fig.text(0.04, 0.5, 'Firing Rates $r_i(t)$' , ha='center', va='center',
             rotation='vertical', fontsize=pa['text_size'], color=txtcolor, fontweight='bold')
    ####### Constant Synapse section 
    ax2 = fig.add_subplot(212)
    t1 =ax2.set_title('Constant Random Synapse', fontsize =pa['text_size'], fontweight='bold', color='black')
    t1.set_position([.5, 1.12])

    pylab.ylim([0, height])
    pylab.xlim([0, len(frsu[:ms])-1])
    
    #txspots = [xspots[0] , xspots[1], xspots[2]] 
    tyspot = height + .01
    yspot = 0
    currentAxis = plt.gca()
    ax2.set_xlabel('Time (ms) ', fontsize=pa['text_size'], color=txtcolor, fontweight='bold')
    for i in range(len(xspots)):
        currentAxis.add_patch(Rectangle((xspots[i], 0), width, height, facecolor="lightgrey", edgecolor=blue))### add gray bars
        plt.text(txspots[i], tyspot, r'$\hat{s}(t) = $' + str(np.round(dfu['0'][(int(xspots[i]))], 2)), color=blue, fontsize =pa['text_size'], fontweight='bold') ### add text 
        #plt.text(xspots[i], tyspot, r'$s = \sum_{i=0} d_i r_i$' +'=' + str(np.round(dfu.ix(int(xspots[i])), 2)), color=blue, fontsize =pa['text_size'], fontweight='bold') ### add text 
    for i in range(len(frsu.columns)):
        a, = plt.plot(x, frsu[str(i)][:ms],  red, linestyle='--',linewidth=2.0)

    ###### Plastic synapse section
    txspots = [xspots[0], xspots[1]-60, xspots[2]-110]
    ax1 = fig.add_subplot(211)
    t2 = ax1.set_title('Plastic Random Synapse', fontsize = pa['text_size'], fontweight='bold', color='black')
    t2.set_position([.5, 1.14])
    ax1.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
    pylab.ylim([0, height])
    pylab.xlim([0, len(frst[:ms])-1])
    currentAxis = plt.gca()
    for i in range(len(xspots)):
        currentAxis.add_patch(Rectangle((xspots[i], 0), width, height, facecolor="lightgrey", edgecolor=blue)) ### add gray bars
        plt.text(txspots[i], tyspot, r'$\hat{s}(t) = $' +str(np.round(dft['0'][(int(xspots[i]))], 2)), color=blue, fontsize =pa['text_size'], fontweight='bold') ### add text 

        #plt.text(xspots[i], tyspot, r'$s = \sum_{i=0} d_i r_i$' +'=' + str(np.round(dft.ix(int(xspots[i])), 2)), color=blue,fontsize =pa['text_size'], fontweight='bold') ### add text 
    for i in range(len(frst.columns)):
        a, = plt.plot(x,frst[str(i)][:ms], red, linestyle='--', linewidth=2.0)

    ## plot final 
    plt.subplots_adjust(hspace = .3)
    plt.setp(ax1.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax1.get_yticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax2.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax2.get_yticklabels(), fontsize=pa['tick_label_size'])
    if(pa['show']):
        plt.show()
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.eps', dpi=1200)
    plt.close() 
    return True

def plot_pqs_cutaxis(dfs, pa, lim1, lim2):
    means = list()
    errors = list()
    for i in range(len(dfs)):
        means.append(dfs[i].mean(axis=1).as_matrix())
        if(pa['plt_errors']):
            errors.append(stats.sem(dfs[i].as_matrix().transpose()))
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    if(pa['right_margins']):
        fig.subplots_adjust(right=.75)
    ax.set_title(pa['title'], fontsize=pa['title_size'], fontweight='bold')
    ax2.set_xlabel(pa['xlabel'], fontsize=pa['label_size'] , fontweight='bold')
    #ax2.set_ylabel(pa['ylabel'], fontsize=pa['label_size'], fontweight='bold')  ## manually set y label 
    x = list(range(len(dfs[0])))
    if('xspot' in pa):
        xspot = pa['xspot']
    else:
        xspot = 0
    pylab.xlim([0, len(x)-1])
    ax.set_ylim(lim1)
    ax2.set_ylim(lim2)
    if('yadds' not in pa):
        pa['yadds'] = np.repeat(0, len(dfs))
    if('linestyle' not in pa):
        pa['linestyle'] = np.repeat('-', len(dfs))
    elif(type(pa['yadds']) is not list):
        pa['yadds'] = np.repeat(pa['yadds'], len(dfs))
    for i, df in enumerate(dfs):
        ax.plot(x, means[i], color=pa['colors'][i], label = pa['labels'][i], linestyle = pa['linestyle'][i], linewidth=pa['linewidth'])
        ax2.plot(x, means[i], color=pa['colors'][i], label = pa['labels'][i], linestyle = pa['linestyle'][i], linewidth=pa['linewidth'])
        if(pa['plt_errors']):
            ax.fill_between(x, means[i] - errors[i], means[i] + errors[i], color=colorAlpha_to_rgb(pa['colors'][i], alpha=.5))
            ax2.fill_between(x, means[i] - errors[i], means[i] + errors[i], color=colorAlpha_to_rgb(pa['colors'][i], alpha=.5))
        if 'yspots' in pa :
            yspot = pa['yspots'][i]
            
        else:
            if(xspot < len(x)):
                yspot = means[i][xspot]
            else:
                yspot = means[i][len(x)-1]
            yspot += + pa['yadds'][i]
        plt.text(xspot, yspot, pa['labels'][i], color=pa['colors'][i], fontsize=pa['text_size'], fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax2.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax2.get_yticklabels(), fontsize=pa['tick_label_size'])

    ## hide spines between ax and ax2
    ax2.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')
    ax.tick_params(top='off')
    ax2.xaxis.tick_bottom()
    
    ## make break lines
    d = .015 ## how big to make diagonal lines in axes coordinate
    kwargs=dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d,+d), **kwargs)
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    if(not pa['right_margins']):
         plt.tight_layout()
    if(pa['show']):  
        plt.show()
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.pdf')
    
    plt.close()
    return True

def plot_weights(weights_df, pa):
    fig = plt.gcf()
    if(pa['right_margins']):
        fig.subplots_adjust(right=.75)
    ax = fig.add_subplot(111)
    ax.set_title(' ', fontweight='bold')
    ax.set_xlabel(pa['xlabel'], fontsize=pa['label_size'] , fontweight='bold')
    ax.set_ylabel('Synaptic Weights', fontsize=pa['ylabel_size'], fontweight='bold')
    x = list(range(weights_df.shape[0]))
    
    if('ylim' in pa):
       pylab.ylim(pa['ylim'])
    if('xlim' in pa):
        pylab.xlim(pa['xlim'])
    else:
        pylab.xlim([0, len(x)-1])
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])

    for i in range(weights_df.shape[1]):
        plt.plot(x, weights_df[str(i)])
    if(not pa['right_margins']):
         plt.tight_layout()
    if(pa['show']):  
        plt.show()
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.pdf')
    plt.close()
    return True

def plot_hist(hist_dfs, pa, sample_size, sample = True, bins=100, alpha=.3, log=False):
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    if(sample):
        hists = list()
        for i, hdf in enumerate(hist_dfs):
            hists.append(np.random.choice(hdf.values.ravel(), sample_size))
    else:
        hists = list()
        for i, hdf in enumerate(hist_dfs):
            hists.append(hdf.values.ravel())
    for i, h_values in enumerate(hists):
        plt.hist(h_values, bins,color=pa['colors'][i], alpha=alpha, log=log)
    plt.xlabel('Synaptic Weight Updates', fontsize=pa['label_size'], fontweight='bold')
    plt.ylabel('n', fontsize=pa['label_size'], fontweight='bold')
    plt.title('')
    plt.legend(loc='upper middle')
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])
    #ax.set_xticks([-.00003, -.000015, 0])
    #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if(pa['show']):  
        plt.show()
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.pdf')
    
def plot_pqs_LIF(dfs, pa, ms):
    """ plots frac of stim remained results from quantify_sims_multithreading"""
    means = list()
    errors = list()
    for i in range(len(dfs)):
        means.append(dfs[i].mean(axis=1).as_matrix())
        if(pa['plt_errors']):
            errors.append(stats.sem(dfs[i].as_matrix().transpose()))
    fig = plt.gcf()
    if(pa['right_margins']):
            fig.subplots_adjust(right=.75)
    ax = fig.add_subplot(111)
    ax.set_title(' ', fontweight='bold')
    ax.set_xlabel(pa['xlabel'], fontsize=pa['label_size'] , fontweight='bold')
    ax.set_ylabel(pa['ylabel'], fontsize=pa['label_size'], fontweight='bold')
    x = np.linspace(0, ms, len(dfs[0]))
    if('xspot' in pa):
        xspot = pa['xspot']
    else:
        xspot = 0
    pylab.xlim([0, ms-1])
    if('ylim' in pa):
        pylab.ylim(pa['ylim'])
    if('yadds' not in pa):
        pa['yadds'] = np.repeat(0, len(dfs))
    if('linestyle' not in pa):
        pa['linestyle'] = np.repeat('-', len(dfs))
    elif(type(pa['yadds']) is not list):
        pa['yadds'] = np.repeat(pa['yadds'], len(dfs))
    for i, df in enumerate(dfs):
        t, = plt.plot(x, means[i], color=pa['colors'][i], label = pa['labels'][i], linestyle = pa['linestyle'][i], linewidth=pa['linewidth'])
        if(pa['plt_errors']):
            plt.fill_between(x, means[i] - errors[i], means[i] + errors[i], color=colorAlpha_to_rgb(pa['colors'][i], alpha=.5))
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
        fig.savefig('plots/' + pa['save_as'] + '.eps', dpi=1200 )
    plt.close()
    return True

def plot_raster(pa, args):
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ax.set_title(' ')
    ax.set_xlabel('Time (ms)', fontsize=pa['label_size'] , fontweight='bold')
    ax.set_ylabel('Neuron', fontsize=pa['label_size'], fontweight='bold')
    for t in range(args['ms']):
        for ne in range(args['ne']):
            if(args['spikes_e_df'][ne][t]==1):
                plt.vlines(t, ne-.2, ne+.2)
    plt.setp(ax.get_xticklabels(), fontsize=pa['tick_label_size'])
    plt.setp(ax.get_yticklabels(), fontsize=pa['tick_label_size'])
    if(pa['show']):  
        plt.show()
    if(pa['save']):
        fig.savefig('plots/' + pa['save_as'] + '.eps', dpi=1200 )
    plt.close()
    return True

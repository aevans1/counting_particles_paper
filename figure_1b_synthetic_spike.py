import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import utils

def main():

    # Can uncomment this for nicer, slower figs, if tex is loaded
    # note: there may be font size errors with this flag off
    #mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.usetex'] = False
    
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'DejaVu Sans'
    mpl.rcParams['axes.titlesize'] = 28
    mpl.rcParams['axes.labelsize'] = 28
    mpl.rcParams['xtick.labelsize'] = 28
    mpl.rcParams['ytick.labelsize'] = 28
    mpl.rcParams['legend.fontsize'] = 28
    mpl.rcParams['lines.linewidth'] = 5 
    mpl.rcParams['lines.markersize'] = 6
    mpl.rcParams['axes.linewidth'] = 4
    mpl.rcParams['xtick.major.width'] = 4
    mpl.rcParams['ytick.major.width'] = 4
    mpl.rcParams['xtick.major.size'] = 12
    mpl.rcParams['ytick.major.size'] = 12
    mpl.rcParams['legend.facecolor'] = 'white'
    mpl.rcParams['legend.edgecolor'] = 'white'
    mpl.rcParams['legend.frameon'] = False
    plt.style.use("seaborn-v0_8-colorblind")
    #plt.style.use("tableau-colorblind10")
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    xticks_SNR = [10**-1, 10**-4, 10**-6]
    xticks_noise_level = [10**-1, 10**1, 10**3]
   
    plot_folder = "figures/"
    data_folder = "data/spike_synthetic/"

    noise_levels = utils.pickle_load(f"{data_folder}/noise_levels.pkl")
    plot_idx = 8
    noise_levels = noise_levels[:plot_idx]
        
    # Load in stats
    fname = f"{data_folder}/pops_errors.pkl"
    pops_errors = utils.pickle_load(fname)
    deconv_weights = pops_errors["deconv_weights"][:plot_idx, :]
    em_weights = pops_errors["em_weights"][:plot_idx, :]
    soft_weights = pops_errors["soft_weights"][:plot_idx, :]
    hard_weights = pops_errors["hard_weights"][:plot_idx, :]

    # Load in cryosparc stuff
    cryosparc_true_volumes_populations = 0.01*np.load(f"{data_folder}/cryosparc_classification_true_volumes_results.npy")[:plot_idx, :]

    # Plot 
    figname = "spike_synthetic_all_populations"
    plt.figure(figsize=(10, 6))
    plt.semilogx(noise_levels, hard_weights[:, 0], label='Hard Assign', color=colors[0], marker='o',  linestyle='solid')
    plt.semilogx(noise_levels, soft_weights[:, 0], label='Soft Assign', color=colors[1], marker='o',  linestyle='solid')
    plt.semilogx(noise_levels, em_weights[:, 0], label='Ensemble Reweight', color=colors[2], marker='o',  linestyle='solid')
    plt.semilogx(noise_levels, deconv_weights[:, 0], label='Deconvolve', color=colors[3], marker='o',  linestyle='solid')
    plt.semilogx(noise_levels, cryosparc_true_volumes_populations[:, 0], label='3D Classification', color=colors[4], marker='o', linestyle='solid')

    plt.hlines(y=0.8, xmin=noise_levels[0], xmax=noise_levels[-1], label="True % Population", linestyle="--", color="k")
    plt.yticks([0.5, 0.65, 0.8])
    plt.xlabel('Noise Level')
    plt.ylabel('% Population in state 1')
    plt.title("Estimated Population in State 1")
    plt.ylim([0.48, 0.85])
        
    # reordering the labels 
    handles, labels = plt.gca().get_legend_handles_labels() 
  
    # specify order 
    order = [5, 0, 1, 2, 3, 4] 
  
    # pass handle & labels lists along with order as below 
    plt.legend([handles[i] for i in order], [labels[i] for i in order])  
    plt.tight_layout()
    plt.savefig(f"{plot_folder}/{figname}.png", dpi=600)
    
if __name__ == '__main__':
    main()
    print("Done")
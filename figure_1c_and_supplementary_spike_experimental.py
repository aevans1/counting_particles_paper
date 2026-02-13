import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import sys
import scipy
import matplotlib as mpl

from matplotlib.transforms import ScaledTranslation



def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'DejaVu Sans'
    mpl.rcParams['axes.titlesize'] = 28
    mpl.rcParams['axes.labelsize'] = 28
    mpl.rcParams['xtick.labelsize'] = 28
    mpl.rcParams['ytick.labelsize'] = 28
    mpl.rcParams['legend.fontsize'] = 20
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
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
   
    plot_folder = "figures/"
    data_path = "data/"
    
    ## POSE EXPERIMENT 
    noise_levels = [2, 4, 6, 8, 10]
    
    # Load in stats
    filename = data_path + "rotation_experiment_weights.pkl"
    with open(filename, 'rb') as f:
        file = pickle.load(f)     
    em_weights = file["em_weights"]
    soft_weights = file["soft_weights"]
    hard_weights = file["hard_weights"]
    deconv_weights = file["deconv_weights"]
    
    # Hard-coded values taken from cryosparc jobs 
    csparc_weights = np.zeros((5, 2))
    csparc_weights[:, 0] = 0.01*np.array([75.3, 59.7, 51.3, 50.9, 50.2])
    csparc_weights[:, 1] = 1 - csparc_weights[:, 0]

    figname = "spike_experimental_rotation_experiment" 
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, hard_weights[:, 0], label='Hard Assign', color=colors[0], marker='o',  linestyle='solid')
    plt.plot(noise_levels, soft_weights[:, 0], label='Soft Assign', color=colors[1], marker='o',  linestyle='solid')
    plt.plot(noise_levels, em_weights[:, 0], label='Ensemble Reweight', color=colors[2], marker='o',  linestyle='solid')
    plt.plot(noise_levels, deconv_weights[:, 0], label='Deconvolve', color=colors[3], marker='o',  linestyle='solid')
    plt.plot(noise_levels, csparc_weights[:, 0], label='3D Classification', color=colors[4], marker='o',  linestyle='solid')
    plt.hlines(y=0.8, xmin=noise_levels[0], xmax=noise_levels[-1], label="True % Population", linestyle="--", color="k")
    plt.ylim([0.48, 0.82])
    plt.legend()
    plt.title("Rotation Misspecification")
    plt.xlabel("Magnitude of noise added (degrees)")
    plt.tight_layout()
    plt.savefig(f"figures/{figname}" + ".png", dpi=600)
    plt.savefig(f"figures/{figname}" + ".pdf", dpi=600)


    # SHIFTS EXPERIMENT
    # Load in stats
    noise_levels = [3.0, 3.25, 3.5, 3.75, 4.0]

    filename = data_path + "shift_experiment_weights.pkl"
    with open(filename, 'rb') as f:
        file = pickle.load(f)     
    em_weights = file["em_weights"]
    soft_weights = file["soft_weights"]
    hard_weights = file["hard_weights"]
    deconv_weights = file["deconv_weights"]

    # Hard-coded values taken from cryosparc jobs 
    csparc_weights = np.zeros((5, 2))
    csparc_weights[:, 0] = 0.01*np.array([73.4, 69.9, 62.4, 51.4, 50.8])
    csparc_weights[:, 1] = 1 - csparc_weights[:, 0]

    figname = "spike_experimental_shift_experiment" 
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, hard_weights[:, 0], label='Hard Assign', color=colors[0], marker='o',  linestyle='solid')
    plt.plot(noise_levels, soft_weights[:, 0], label='Soft Assign', color=colors[1], marker='o',  linestyle='solid')
    plt.plot(noise_levels, em_weights[:, 0], label='Ensemble Reweight', color=colors[2], marker='o',  linestyle='solid')
    plt.plot(noise_levels, deconv_weights[:, 0], label='Deconvolve', color=colors[3], marker='o',  linestyle='solid')
    plt.plot(noise_levels, csparc_weights[:, 0], label='3D Classification', color=colors[4], marker='o',  linestyle='solid')
    plt.hlines(y=0.8, xmin=noise_levels[0], xmax=noise_levels[-1], label="True % Population", linestyle="--", color="k")
    plt.xticks(noise_levels)
    plt.ylim([0.48, 0.82])
    plt.legend()
    plt.title("Shift Misspecification")
    plt.xlabel("Magnitude of noise added (angstroms)")
    plt.tight_layout()
    plt.savefig(f"figures/{figname}" + ".png", dpi=600)
    plt.savefig(f"figures/{figname}" + ".pdf", dpi=600)


    # NOISE TO IMAGES EXPERIMENT
    # Load in stats
    filename = data_path + "ground_truth_experiment_weights.pkl"
    with open(filename, 'rb') as f:
        file = pickle.load(f)     
    em_weights = file["em_weights"]
    soft_weights = file["soft_weights"]
    hard_weights = file["hard_weights"]
    deconv_weights = file["deconv_weights"]

    hard_assignment = [hard_weights[0, 0], hard_weights[1, 0]]
    soft_assignment = [soft_weights[0, 0], soft_weights[1, 0]]
    ensemble_reweighting = [em_weights[0, 0], em_weights[1, 0]]
    deconvolution = [deconv_weights[0, 0], deconv_weights[1, 0]]
    three_dee_classification_gt = [0.76, 0.608]

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 6))
    barWidth = 0.01
    br1 = 0
    br2 = br1 + barWidth
    br3 = br2 + barWidth
    br4 = br3 + barWidth
    br5 = br4 + barWidth

    ax1.bar(br1, hard_assignment[0], color =colors[0], width = barWidth, align = 'center', 
    edgecolor ='black', label ='Hard Assign')
    ax1.bar(br2, soft_assignment[0], color =colors[1], width = barWidth, align = 'center', 
    edgecolor ='black', label ='Soft Assign')
    ax1.bar(br3, ensemble_reweighting[0], color =colors[2], width = barWidth, align = 'center', 
    edgecolor ='black', label ='Ensemble Reweight')
    ax1.bar(br4, deconvolution[0], color =colors[3], width = barWidth, align = 'center', 
    edgecolor ='black', label ='Deconvolve')
    ax1.bar(br5, three_dee_classification_gt[0], color = colors[4], width = barWidth, align = 'center', 
    edgecolor ='black', label ='3D Classification')

    ax1.set_yticks([0.5, 0.65, 0.8])
    ax1.set_xticks([], [])
    ax1.hlines(y=0.8, xmin=br1-barWidth, xmax=br5+barWidth, label='\% population, EMD-50421',linestyle="--", color="k")
    ax1.hlines(y=0.8, xmin=br1-barWidth, xmax=br5+barWidth, label='True',linestyle="--", color="k")
    ax1.set_ylim([0.48, 0.85])
    ax1.set_xticks([br3], 
            ['original'])
    ax1.set_xlim([br1-barWidth, br5+barWidth])
    
    ax2.bar(br1, hard_assignment[1], color =colors[0], width = barWidth, align = 'center', 
    edgecolor ='black', label ='Hard Assign')
    ax2.bar(br2, soft_assignment[1], color =colors[1], width = barWidth, align = 'center', 
    edgecolor ='black', label ='Soft Assign')
    ax2.bar(br3, ensemble_reweighting[1], color =colors[2], width = barWidth, align = 'center', 
    edgecolor ='black', label ='Ensemble Reweight')
    ax2.bar(br4, deconvolution[1], color =colors[3], width = barWidth, align = 'center', 
    edgecolor ='black', label ='Deconvolve')
    ax2.bar(br5, three_dee_classification_gt[1], color = colors[4], width = barWidth, align = 'center', 
    edgecolor ='black', label ='3D Classification')

    figname = "spike_experimental_split_plot" 
    ax2.set_yticks([], [])
    #ax2.set_xticks([], [])
    ax2.hlines(y=0.8, xmin=br1-barWidth, xmax=br5+barWidth, label='True',linestyle="--", color="k")
    ax2.set_ylim([0.48, 0.85])
    ax2.set_xlim([br1-barWidth, br5+barWidth])
    ax2.set_xticks([br3], ['noise added'])
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=28)
    
    plt.subplots_adjust(hspace=1.0)
    plt.tight_layout()
    plt.savefig(f"figures/{figname}_subfigs" + ".png", dpi=600)

    plt.show()

if __name__ == '__main__':
    main()
    print("Done")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import matplotlib as mpl

from matplotlib.transforms import ScaledTranslation


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

    # This needs to match the noise levels used in generating the data. 
    noise_levels = np.logspace(-1, 4, 10)  

    plot_idx = 8
    noise_levels = noise_levels[:plot_idx]
        
    # Load in stats
    file = open(data_folder + '/' + 'pops_errors.pkl','rb')
    pops_errors = pickle.load(file)
    error_observed = pops_errors["error_observed"][:plot_idx]
    error_predicted = pops_errors["error_predicted"][:plot_idx]
    deconvolve_pop = pops_errors["deconvolve_pop"][:plot_idx, :]
    observed_pop_soft = pops_errors["observed_pop_soft"][:plot_idx, :]
    observed_pop = pops_errors["observed_pop"][:plot_idx, :]
    file.close()

    file = open(data_folder + '/' + 'extra_stats.pkl','rb')
    extra_stats = pickle.load(file) 
    deconvolve_observed = extra_stats["deconvolve_observed"][:plot_idx]
    reweight_observed = extra_stats["reweight_observed"][:plot_idx]
    bayes_observed = extra_stats["bayes_observed"][:plot_idx]
    reweight_pop = extra_stats["reweight_pop"][:plot_idx, :]
    file.close()

    # Load in cryosparc stuff
    cryosparc_true_volumes_populations = 0.01*np.load(f"{data_folder}/cryosparc_classification_true_volumes_results.npy")[:plot_idx, :]
    cryosparc_populations = 0.01*np.load(f"{data_folder}/cryosparc_classification_results.npy")[:plot_idx, :]

    # Plot 
    figname = "classification_all_methods"

    plt.figure(figsize=(10, 6))
    plt.semilogx(noise_levels, error_predicted, label='Hard Assign, Analytical', marker='o', color='k', linestyle='solid')
    plt.semilogx(noise_levels, error_observed, label='Hard Assign',  marker='o', color=colors[0], linestyle='solid')
    plt.semilogx(noise_levels, bayes_observed, label='Bayes Optimal', marker='o', color=colors[4],  linestyle='solid')
    plt.semilogx(noise_levels, deconvolve_observed, label='Deconvolve', marker='o', color=colors[3],   linestyle='solid')
    plt.semilogx(noise_levels, reweight_observed, label='Ensemble Reweight', marker='o', color=colors[2],  linestyle='solid')
    plt.yticks([0.1, 0.3, 0.5])
    plt.xlabel('Noise Level')
    plt.ylabel('Misclassification Rate')
    plt.title('Misclassification Rate vs. Noise Level')
    leg = plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder + '/' + figname + '.png', dpi=600)

    figname = "populations_all_methods"
    plt.figure(figsize=(10, 6))
    plt.semilogx(noise_levels, observed_pop[:, 0], label='Hard Assign', color=colors[0], marker='o',  linestyle='solid')
    plt.semilogx(noise_levels, observed_pop_soft[:, 0], label='Soft Assign', color=colors[1], marker='o',  linestyle='solid')
    plt.semilogx(noise_levels, reweight_pop[:, 0], label='Ensemble Reweight', color=colors[2], marker='o',  linestyle='solid')
    plt.semilogx(noise_levels, deconvolve_pop[:, 0], label='Deconvolve', color=colors[3], marker='o',  linestyle='solid')
    plt.semilogx(noise_levels, cryosparc_true_volumes_populations[:, 0], label='3D Classification', color=colors[4], marker='o', linestyle='solid')

    plt.hlines(y=0.8, xmin=noise_levels[0], xmax=noise_levels[-1], label="True % Population", linestyle="--", color="k")
    plt.yticks([0.5, 0.65, 0.8])
    plt.xlabel('Noise Level')
    plt.ylabel('% Population in state 1')
    plt.title("Estimated Population in State 1")
    figname = figname + "_with_labels"
    plt.ylim([0.48, 0.85])
        
    # reordering the labels 
    handles, labels = plt.gca().get_legend_handles_labels() 
  
    # specify order 
    order = [5, 0, 1, 2, 3, 4] 
  
    # pass handle & labels lists along with order as below 
    plt.legend([handles[i] for i in order], [labels[i] for i in order])  
    plt.tight_layout()
    plt.savefig(plot_folder + '/' + figname + '.png', dpi=600)
    
    # plot analytical and observed misclassificaiton errors
    figname = "error_predicted_observed" 
    plt.figure(figsize=(10, 6))
    plt.semilogx(noise_levels, error_predicted, label='Analytical', color='k', marker='o')
    plt.semilogx(noise_levels, error_observed, label='Observed', color=colors[0], marker='s')
    plt.yticks([0.1, 0.3, 0.5])
    plt.xlabel('Noise Level')
    plt.ylabel('Misclassification Rate')
    plt.title('Misclassification Rate vs. Noise Level')
    figname = figname + "_with_labels"
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder + '/' + figname + '.png')
    plt.show()

if __name__ == '__main__':
    main()
    print("Done")
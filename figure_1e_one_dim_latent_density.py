import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import vonmises
import seaborn as sns
sns.set_style("ticks")
sns.color_palette("colorblind")

plt.style.use("seaborn-v0_8-colorblind")
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Can uncomment this for nicer, slower figs, if tex is loaded
# note: there may be font errors with this flag off
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.usetex'] = False

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
#mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['legend.facecolor'] = 'white'
mpl.rcParams['legend.edgecolor'] = 'white'
mpl.rcParams['legend.frameon'] = False


def main():

    plot_folder = "figures/"
    data_path = "data/igg/"

        
    figname = "igg_1d_latent_densities"

    ## define density that volumes were resampled from
    def p(x):
        means = [np.pi/2, np.pi, 3*np.pi/2]
        kappas =  [6.0, 6.0, 6.0]
        weights = np.array([2.0, 1.0, 2.0])
        weights /= sum(weights)  
        val = 0
        for i in range(3): 
            val += weights[i]*vonmises.pdf(x, loc=means[i], kappa=kappas[i])
        return val
    x = np.linspace(0, 360, 100)
    y = p(x*(2*np.pi / 360))
    y /= (np.sum(y))

    # Load 1d density from histogramming of principal components
    raw_density_at_zs_gt = np.load(f"{data_path}/1d_density_raw.npy") 

    # Load 1d density from deconvolving of principal components raw density
    k =  3 # specify which density to get, in this case its the "elbow" density
    density_at_zs_gt = np.load(f"{data_path}/1d_density_reg_{k}.npy") 

    # Load 1d density from ensemble reweighting given the ground truth volumes (reference)
    reweighted = np.load(f"{data_path}/1d_reweight_density_reference.npy")


    plt.figure(figsize=(10,6)) 
    plt.plot(x, raw_density_at_zs_gt, label="Histogram")
    plt.plot(x, reweighted, label="Ensemble Reweight", color=colors[2])
    plt.plot(x, density_at_zs_gt, label="Deconvolve", color=colors[3])
    plt.plot(x, y, label="True", color='k', linestyle='dashed')
    plt.xlim([-0.1, 360.1])

    plt.xlabel(r"Dihedral Angle ($\circ$)")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(0.3, 0.62))    
    
    #specify number of ticks on y-axis
    plt.locator_params(axis='y', nbins=5)
    
    plt.xticks([0,180,360])
    plt.yticks([0, 0.01, 0.02])
    plt.savefig(f"{plot_folder}/{figname}.png", dpi=600, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()

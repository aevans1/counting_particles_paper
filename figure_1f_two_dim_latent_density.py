import numpy as np
import matplotlib.pyplot as plt

import utils

def main():

    plot_folder = "figures/"
    data_path = "data/igg/"

    cmap="inferno" 
    
    # Get embedding from ground truth volumes
    zs_gt_fname = f"{data_path}/embedded_gt_volumes_zdim2.npy"
    zs_gt = np.load(zs_gt_fname)   

    # Get density and PC bounds
    density_file = np.load(f"{data_path}/raw_density.npz")
    density = density_file['density']
    density_bounds = density_file['latent_space_bounds']

    # Reshape embedded values to the PC grid
    zs_gt_grid = utils.zs_to_grid(zs_gt, density_bounds, density.shape[0])

    # Plot raw density with embedded volumes visualized
    figname = "igg_raw_density_gt_vols"
    fig = plt.subplots()
    axs = plt.gca()
    to_plot = density
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    axs.xaxis.set_ticks_position('none') 
    axs.yaxis.set_ticks_position('none') 
    axs.imshow(to_plot.T, cmap=cmap)
    axs.scatter(zs_gt_grid[:, 0], zs_gt_grid[:, 1], c="k", edgecolors='w', s=10, linewidths=0.5)
    axs.set_xlabel("PC 0")
    axs.set_ylabel(f"PC {1}")
    plt.savefig(f"{plot_folder}/{figname}.png", dpi=300)


    k = 3
    # Load pre-computed density info
    density_file = np.load(f"{data_path}/deconv_density_{k}.npz")
    density = density_file['density']
    density_bounds = density_file['latent_space_bounds']

    # Reshape embedded values to the PC grid
    zs_gt_grid = utils.zs_to_grid(zs_gt, density_bounds, density.shape[0])

    figname = "igg_deconv_density_gt_vols"
    fig = plt.subplots()
    axs = plt.gca()
    to_plot = density
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    axs.xaxis.set_ticks_position('none') 
    axs.yaxis.set_ticks_position('none') 
    axs.imshow(to_plot.T, cmap=cmap)
    axs.scatter(zs_gt_grid[:, 0], zs_gt_grid[:, 1], c="k", edgecolors='w', s=10, linewidths=0.5)
    axs.set_xlabel("PC 0")
    axs.set_ylabel(f"PC {1}")
    plt.savefig(f"{plot_folder}/{figname}.png", dpi=300)


if __name__ == "__main__":
    main()


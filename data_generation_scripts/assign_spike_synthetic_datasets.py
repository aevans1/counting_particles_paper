import numpy as np
import jax.numpy as jnp

from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import image_assignment, noise
from recovar import simulator, image_assignment, noise, dataset
import pickle
import argparse

from importlib import reload
reload(simulator)

def main():

    # script is written to work on one "noise_level" dataset at a time, bash script calls all needed noise_levels
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--noise_level_idx", type=int)

    args = parser.parse_args()
    output_folder = args.output_folder
    idx = args.noise_level_idx

    disc_type_infer = 'cubic'
    grid_size = 256
 
    # Load in things
    file = open(output_folder + '/' + 'noise_levels.pkl','rb')
    noise_levels = pickle.load(file)
    noise_level = noise_levels[idx]

    dataset_folder = output_folder + '/' + f'dataset{idx}/'
    print(f"Running noise level {idx} of {len(noise_levels)-1}, noise variance added is {noise_level}") 
        
    # Load in simulation data
    file = open(dataset_folder + '/' + 'sim_info.pkl','rb')
    sim_info = pickle.load(file)
    file.close()  

    # Load datasets and volumes
    # Volumes are scaled so that images are normalized. So they have a slightly different scale for each dataset.
    print(sim_info['volumes_path_root']) 

    volumes = simulator.load_volumes_from_folder(sim_info['volumes_path_root'], sim_info['grid_size'] , sim_info['trailing_zero_format_in_vol_name'], normalize=False )
    gt_volumes = volumes * sim_info['scale_vol']
        
    dataset_options = dataset.get_default_dataset_option()
    dataset_options['particles_file'] = dataset_folder + f'particles.{grid_size}.mrcs'
    dataset_options['ctf_file'] = dataset_folder + f'ctf.pkl'
    dataset_options['poses_file'] = dataset_folder + f'poses.pkl'
    cryo = dataset.load_dataset_from_dict(dataset_options, lazy = False)
        
    # Compute likelihoods
    batch_size = 100
    image_cov_noise = np.asarray(noise.make_radial_noise(sim_info['noise_variance'], cryo.image_shape))

    # transforming image assignment output to log likelihoods
    log_likelihoods = -0.5*image_assignment.compute_image_assignment(cryo, gt_volumes,  image_cov_noise, batch_size, disc_type = disc_type_infer).T

    # compute analytical error rate for assignment
    error_predicted = image_assignment.estimate_false_positive_rate(cryo, gt_volumes,  image_cov_noise, batch_size, disc_type = disc_type_infer)

    # Compute hard assignments, hard assignment uncertainties
    true_assignments = sim_info['image_assignment']
    hard_assignments = jnp.argmax(log_likelihoods, axis = 1)
    observed_pop = np.array([np.mean(hard_assignments==0), np.mean(hard_assignments==1)])

    # Dump results to file
    likelihoods_assignments = { 'log_likelihoods': log_likelihoods, 
              'hard_assignments' : hard_assignments,
              'true_assignments' : sim_info['image_assignment'],
              'error_predicted': error_predicted 
              } 
    recovar.utils.pickle_dump(likelihoods_assignments, dataset_folder + '/' + 'likelihoods_assignments.pkl')
    recovar.utils.pickle_dump({'observed_pop' : observed_pop}, \
                            dataset_folder + '/' + 'pops_errors.pkl') 

if __name__ == '__main__':
    main()
    print("Done")

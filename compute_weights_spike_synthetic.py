import numpy as np
import jax.numpy as jnp

import utils

def main():
    
    # Load in things
    output_folder = "data/spike_synthetic"
    fname = output_folder + '/' + 'noise_levels.pkl'
    noise_levels = utils.pickle_load(fname)

    volume_distribution = np.array([0.8, 0.2])
   
    em_weights = np.zeros((len(noise_levels), 2))
    deconv_weights = np.zeros((noise_levels.size, 2))
    hard_weights = np.zeros((noise_levels.size, 2))
    soft_weights = np.zeros((noise_levels.size, 2))

    for idx, noise_level in enumerate(noise_levels):

        print(f"Starting at noise level {idx} of {len(noise_levels)}") 

        # Load in stats
        fname = output_folder + '/' + f'likelihoods_assignments_dataset{idx}.pkl'
        likelihoods_assignments = utils.pickle_load(fname)

        log_likelihoods = likelihoods_assignments['log_likelihoods']
        hard_assignments = likelihoods_assignments['hard_assignments']
        error_predicted = likelihoods_assignments['error_predicted']
        fname = output_folder + '/' + f'likelihoods_assignments_dataset{idx}.pkl'


        num_data, num_nodes = log_likelihoods.shape
        print(f"number of images used: {num_data}")
        print(f"number of structures used: {num_nodes}")
        
        # Ensemble reweighting
        #NOTE: here we are setting a strict tolerance, as for two distinct structures there is less chance of overfitting 
        em_weights[idx, :] = utils.multiplicative_gradient(log_likelihoods, tol=1e-8)

        # soft classification
        soft_weights[idx, :] = utils.multiplicative_gradient(log_likelihoods, max_iterations=1, tol=-1)

        # hard classification
        classified = jnp.argmax(log_likelihoods, axis=1)
        hard_weights[idx, :] = np.zeros(2)
        hard_weights[idx, 1] = jnp.mean(classified)
        hard_weights[idx, 0] = 1 - hard_weights[idx, 1]

        # Deconvolution
        _ , deconv_weights[idx, :], _ = utils.deconvolve_assignments(hard_assignments, error_predicted)
        
        print('hard weights:', hard_weights[idx])
        print('soft weights:', soft_weights[idx])
        print('em weights:', em_weights[idx, :])
        print('deconv weights:', deconv_weights[idx, :])

    ## Dump results to file
    pops_errors = {'deconv_weights' : deconv_weights, \
                    'soft_weights': soft_weights, \
                    'hard_weights' : hard_weights, \
                    'em_weights' : em_weights, \
                    }
    fname = output_folder + '/' + 'pops_errors.pkl'
    utils.pickle_dump(pops_errors, fname) 


if __name__ == '__main__':
    main()
    print("Done")

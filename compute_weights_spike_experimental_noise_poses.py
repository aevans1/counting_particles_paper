import numpy as np
import jax.numpy as jnp
from utils import deconvolve_assignments, multiplicative_gradient

def main():

    data_names = ["noisy_poses_magn_2", "noisy_poses_magn_4", "noisy_poses_magn_6", "noisy_poses_magn_8", "noisy_poses_magn_10"]

    my_dict = {}
    em_weights = np.zeros((len(data_names), 2))
    soft_weights = np.zeros((len(data_names), 2))
    hard_weights = np.zeros((len(data_names), 2))
    deconv_weights = np.zeros((len(data_names), 2))

    for idx in range(len(data_names)): 
        data_name = data_names[idx]
        volume_name ="ground_truth_recovar_filtered_aligned"
    
        path = "data/"

        log_likelihood = jnp.load(path + f"log_likelihoods_recovar_{data_name}_{volume_name}.npy")
        error_predicted = jnp.load(path + f"error_predicted_recovar_{data_name}_{volume_name}.npy")
        log_likelihood = log_likelihood - jnp.max(log_likelihood, 1)[:, jnp.newaxis]

        num_data, num_nodes = log_likelihood.shape
        print(f"number of images used: {num_data}")
        print(f"number of structures used: {num_nodes}")
        
        # solve with expectation maximization
        em_weights[idx, :] = multiplicative_gradient(log_likelihood)

        # soft classification
        soft_weights[idx, :] = multiplicative_gradient(log_likelihood, max_iterations=1)

        # hard classification
        classified = jnp.argmax(log_likelihood, axis=1)
        hard_weights[idx, :] = np.zeros(2)
        hard_weights[idx, 1] = jnp.mean(classified)
        hard_weights[idx, 0] = 1 - hard_weights[idx, 1]
        
        # Deconvolution
        _ , deconv_weights[idx, :], _ = deconvolve_assignments(classified, error_predicted)
        
    my_dict = {"em_weights":em_weights, "deconv_weights":deconv_weights, "soft_weights":soft_weights, "hard_weights": hard_weights}
    fname = path + "rotation_experiment_weights.pkl"
        
    import pickle
    with open(fname, "wb") as f:
        pickle.dump(my_dict, f)

    # now, shifts
    data_names = ["noisy_shifts_level_0", "noisy_shifts_level_1", "noisy_shifts_level_2", "noisy_shifts_level_3", "noisy_shifts_level_4"]
    my_dict = {}
    em_weights = np.zeros((len(data_names), 2))
    soft_weights = np.zeros((len(data_names), 2))
    hard_weights = np.zeros((len(data_names), 2))
    deconv_weights = np.zeros((len(data_names), 2))

    for idx in range(len(data_names)): 
        data_name = data_names[idx]
        volume_name ="ground_truth_recovar_filtered_aligned"
    
        path = "/mnt/home/levans/Projects/Bad_Histogram/spike_real/data/log_likelihoods_recovar/"

        log_likelihood = jnp.load(path + f"log_likelihoods_recovar_{data_name}_{volume_name}.npy")
        error_predicted = jnp.load(path + f"error_predicted_recovar_{data_name}_{volume_name}.npy")
        log_likelihood = log_likelihood - jnp.max(log_likelihood, 1)[:, jnp.newaxis]

        # solve with expectation maximization (em)
        em_weights[idx, :] = multiplicative_gradient(log_likelihood)

        # soft classification
        soft_weights[idx, :] = multiplicative_gradient(log_likelihood, max_iterations=1)

        # hard classification
        classified = jnp.argmax(log_likelihood, axis=1)
        hard_weights[idx, :] = np.zeros(2)
        hard_weights[idx, 1] = jnp.mean(classified)
        hard_weights[idx, 0] = 1 - hard_weights[idx, 1]
        
        # Deconvolution
        _, deconv_weights[idx, :], _ = deconvolve_assignments(classified, error_predicted)

    my_dict = {"em_weights":em_weights, "deconv_weights":deconv_weights, "soft_weights":soft_weights, "hard_weights": hard_weights}
    fname = path + "shift_experiment_weights.pkl"
    
    import pickle
    with open(fname, "wb") as f:
        pickle.dump(my_dict, f)


if __name__ == "__main__":
    main()

import numpy as np
import jax.numpy as jnp
import tabulate

from utils import deconvolve_assignments, multiplicative_gradient

def main():
    
    data_names = ["ground_truth_poses_80_20", "noisy_ground_truth_poses_80_20"]
    volume_name ="ground_truth_recovar_filtered_aligned"
    
    em_weights = np.zeros((2,2))
    hard_weights = np.zeros((2,2))
    soft_weights = np.zeros((2,2))
    deconv_weights = np.zeros((2,2))
    
    for idx, data_name in enumerate(data_names):
        print(idx)
        path = "data/"

        log_likelihood = np.load(path + f"log_likelihoods_recovar_{data_name}_{volume_name}.npy")
        error_predicted = np.load(path + f"error_predicted_recovar_{data_name}_{volume_name}.npy")
        
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
        _, deconv_weights[idx, :], _ = deconvolve_assignments(classified, error_predicted)

        my_table = [["hard classification", hard_weights[idx, 0], hard_weights[idx, 1]],
                    ["soft classification", soft_weights[idx, 0], soft_weights[idx, 1]],
                    ["ensemble reweighting", em_weights[idx, 0], em_weights[idx, 1]],
                    ["deconvolution", deconv_weights[idx, 0], deconv_weights[idx, 1]]
                    ]
        table = tabulate.tabulate(
                    my_table, 
                    headers=["Method", "Population 1", "Population 2"], 
                    tablefmt="grid"
                    )
        print(table)

    # TODO: import directly from cryosparc metadata
    three_dee_classification = [0.76, 0.608]

    my_dict = {"em_weights":em_weights, "deconv_weights":deconv_weights, "soft_weights":soft_weights, "hard_weights": hard_weights,
    "three_dee_classification":three_dee_classification}
    fname = path + "ground_truth_experiment_weights.pkl"
        
    import pickle
    with open(fname, "wb") as f:
        pickle.dump(my_dict, f)


if __name__ == "__main__":
    main()

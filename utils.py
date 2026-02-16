import numpy as np
import jax.numpy as jnp
import jax
import cvxpy as cp
import pickle


def deconvolve_assignments(assignments, error_predicted):
    observed_pop = np.array([np.mean(assignments==0), np.mean(assignments==1)])
    deconvolve_matrix = np.array( [ [1- error_predicted, error_predicted], [error_predicted, 1- error_predicted] ])
    
    # Run constrained optimization 
    x = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(deconvolve_matrix @ x - observed_pop)) 
    constraints = [0 <= x, x<= 1, sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    deconvolve_pop = x.value 
    return observed_pop, deconvolve_pop, deconvolve_matrix


@jax.jit
def grad_log_prob(weights,likelihood):
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.
    """
    aux =  jnp.sum(likelihood*weights, axis=1)
    grad =  jnp.mean((likelihood) / aux[:, np.newaxis], axis=0)
    return grad


@jax.jit
def update_weights(weights, grad):
    weights = weights*grad
    return weights


def multiplicative_gradient(
    log_likelihood,
    tol=1e-3,
    max_iterations=10000
):

    num_images, num_structures = log_likelihood.shape

    # Initialize Weights
    weights = (1/num_structures)*jnp.ones(num_structures)

    # Normalizing log likelihood by row for stability, the gradient is invariant to this rescaling
    log_likelihood = log_likelihood - jnp.max(log_likelihood, 1)[:, jnp.newaxis]
    
    # Note: this exponentiation cannot happen without previous renormalizing. Re: softmax 
    likelihood = jnp.exp(log_likelihood)
    for k in range(max_iterations):

        # Update weights
        grad = grad_log_prob(weights, likelihood)   
        weights = update_weights(weights, grad)

        # Check stopping criterion
        gap = jnp.max(grad) - 1
        if k % 1000 == 0: 
            print(k)
            print(gap) 

        if gap < tol:
            print(f"number of iterations: {k}")
            print("exiting!")
            break
    return weights


def pickle_dump(object, file):
    with open(file, "wb") as f:
        pickle.dump(object, f)


def pickle_load(file):
    with open(file, "rb") as f:
        return pickle.load(f)

################################################################
# NOTE: functions below are utils from recovar software, latent_density.py
################################################################
def grid_to_pca_coord(v, bounds, num_points):
    x =  v * ( bounds[:,1]  - bounds[:,0] ) / (num_points - 1)  + bounds[:,0]
    return x

def pca_coord_to_grid(x, bounds, num_points, to_int = False):
    v =  (x - bounds[:,0] ) / ( bounds[:,1]  - bounds[:,0] ) * (num_points - 1)    
    if to_int:
        return np.round(v).astype(int)   
    else:
        return v

def get_grid_to_z(bounds, num_points ):
    def grid_to_z(x):
        return grid_to_pca_coord(x, bounds = bounds, num_points = num_points)        
    return grid_to_z

def get_z_to_grid(bounds, num_points ):
    def z_to_grid(x, to_int = False):
        return pca_coord_to_grid(x, bounds = bounds, num_points = num_points, to_int = to_int)        
    return z_to_grid

def get_grid_z_mappings(bounds, num_points):
    return get_grid_to_z(bounds, num_points ), get_z_to_grid(bounds, num_points )

def zs_to_grid(zs, bounds, num_points):
    _, z_to_grid = get_grid_z_mappings(bounds, num_points = num_points)
    zs_grid = z_to_grid(zs)
    return zs_grid



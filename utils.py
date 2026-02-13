import numpy as np
import jax.numpy as jnp
import jax
import cvxpy as cp


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
    tol=1e-30,
    max_iterations=100000
):

    num_images, num_structures = log_likelihood.shape

    # Initialize Weights
    weights = (1/num_structures)*jnp.ones(num_structures).astype('float64')

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

import numpy as np

def generate_heteroskedastic_data(n, p, hetero_func, beta=None, gamma=None, seed=None, dist=None, type='diagonal'):
    if seed is not None:
        np.random.seed(seed)
    if beta is None:
        beta = np.ones(p) # for effect size of features
    if gamma is None:
        gamma = np.ones(p) # for heteroskedasticity
    if dist is None:
        dist = np.random.normal
    if hetero_func is None:
        raise ValueError("The 'hetero_func' must be provided.")

    X = dist(size=(n, p))
    beta = np.array(beta)
    gamma = np.array(gamma)

    try:
        if type == 'diagonal':
            X_gamma = X @ gamma
            shifted_X_gamma = X_gamma + np.abs(np.min(X_gamma)) # shift to be positive
            variances = np.array(hetero_func(shifted_X_gamma)) # heteroskedasticity function
            truth_var = np.maximum(variances, 1) # ensure variances are at least 1
            # use exact variances for noise power
            errors = dist(np.zeros(n), np.sqrt(truth_var))
        elif type == 'unstructured':
            X_gamma = X @ gamma
            shifted_X_gamma = X_gamma + np.abs(np.min(X_gamma)) # shift to be positive
            sd = np.sqrt(hetero_func(shifted_X_gamma)) # heteroskedasticity function for diagonal
            cov = np.diag(np.maximum(sd**2, 1)) # ensure diagonal is at least 1
            
            # add noise to the covariance matrix
            noise = dist(size=(n, n), scale=0.5)
            noise = noise + noise.T
            np.fill_diagonal(noise, 0) # replace diagonal with zeros

            # add noise to cov matrix
            truth_var = cov + sd[:, np.newaxis] * noise * sd[np.newaxis, :] # cov = cov + sd * noise * sd (scaled by sd)
            
            errors = np.random.multivariate_normal(np.zeros(n), cov=truth_var)
        else:
            raise ValueError(f"Please provide type as 'diagonal' or 'unstructured'. Now got {type}." )
    except KeyError as e:
        raise KeyError(f"An error occurred with the variance function: {e}.")
    
    signal_power = (X @ beta)**2  # size (n,)
    if len(truth_var.shape) == 2: # when multivariate, extract the diagonal
        noise_power = np.diag(truth_var)
    else:
        noise_power = truth_var # else only use 1-d diagonal array
    snr = np.divide(signal_power, noise_power)
    range_snr = np.array([np.min(snr), np.max(snr)])

    # obtain y
    y = X @ beta + errors

    return X, y, truth_var, range_snr

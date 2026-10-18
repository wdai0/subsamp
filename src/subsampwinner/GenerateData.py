"""Legacy heteroskedastic data generator with local randomness and validation."""

import numpy as np
from ._validation import finite_array, positive_integer, validate_seed


def generate_heteroskedastic_data(n, p, hetero_func, beta=None, gamma=None, seed=None, dist=None, type='diagonal'):
    """Generate the historical design/noise recipe without global RNG resets.

The default uses a local RandomState. A custom normal-like ``dist`` callable
owns its own randomness. Variances retain the historical lower bound of one.
The legacy unstructured recipe can be indefinite; reject such draws explicitly.
A different PSD covariance recipe is a separate data-generating-method change.
    """
    positive_integer(n, "n")
    positive_integer(p, "p")
    validate_seed(seed)
    if not callable(hetero_func):
        raise ValueError("hetero_func must be callable")
    if type not in ("diagonal", "unstructured"):
        raise ValueError("type must be 'diagonal' or 'unstructured'")
    rng = np.random.RandomState(seed)
    if dist is None:
        dist = rng.normal
    if not callable(dist):
        raise ValueError("dist must be callable")
    beta = np.ones(p) if beta is None else finite_array(beta, "beta")
    gamma = np.ones(p) if gamma is None else finite_array(gamma, "gamma")
    if beta.shape != (p,) or gamma.shape != (p,):
        raise ValueError("beta and gamma must have length p")
    X = finite_array(dist(size=(n, p)), "generated X")
    if X.shape != (n, p):
        raise ValueError("dist must return the requested design shape")
    signal, X_gamma = X @ beta, X @ gamma
    shifted = X_gamma + abs(np.min(X_gamma))
    variances = finite_array(hetero_func(shifted), "variances")
    if variances.ndim == 0:
        variances = np.full(n, float(variances))
    if variances.shape != (n,) or np.any(variances < 0):
        raise ValueError("hetero_func must return nonnegative variances, scalar or length n")
    if type == 'diagonal':
        truth_var = np.maximum(variances, 1)
        errors = dist(np.zeros(n), np.sqrt(truth_var))
    else:
        sd = np.sqrt(variances)
        noise = finite_array(dist(size=(n, n), scale=.5), "covariance perturbation")
        if noise.shape != (n, n):
            raise ValueError("dist must return the requested covariance shape")
        noise = noise + noise.T
        np.fill_diagonal(noise, 0)
        truth_var = np.diag(np.maximum(variances, 1)) + sd[:, None] * noise * sd[None, :]
        eigenvalues = np.linalg.eigvalsh(truth_var)
        if eigenvalues[0] < 0:
            raise ValueError("legacy unstructured covariance is not positive semidefinite; no data generated")
        errors = rng.multivariate_normal(np.zeros(n), truth_var, check_valid='raise')
    errors = finite_array(errors, "generated errors")
    if errors.shape != (n,):
        raise ValueError("dist must return one error per observation")
    y = finite_array(signal + errors, "generated y")
    noise_power = np.diag(truth_var) if truth_var.ndim == 2 else truth_var
    snr = finite_array(signal**2 / noise_power, "pointwise signal-to-noise ratio")
    return X, y, truth_var, np.array([np.min(snr), np.max(snr)])

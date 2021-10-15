from nrmifactors.priors import NNIGPrior
import jax.numpy as np
from jax import jit, random
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from nrmifactors.priors import NNIGPrior

@jit
def nnig_update(data, mu0, lam, a, b, key):
    ybar = np.mean(data)
    card = data.shape[0]
    post_mean = (mu0 / 25.0 + np.sum(data) / 1.0) / (1.0 / 25.0 + card / 1.0)
    post_var = 1.0 / (1.0 / 25.0 + card / 1.0)
    key, subk1 = random.split(key)
    
    mu = tfd.Normal(post_mean, np.sqrt(post_var)).sample(seed=subk1)
    return np.array([mu, 1.0]), key

    # post_mean = (lam * mu0 + np.sum(data)) / (lam + card)
    # post_lam = lam + card
    # post_shape = a + 0.5 * card
    # post_rate = b + \
    #     0.5 * np.sum((data - ybar)**2) + \
    #     0.5 * lam * card * (mu0 - ybar)**2 / post_lam
    # key, subk1, subk2 = random.split(key, 3)
    # var_post = tfd.InverseGamma(post_shape, post_rate).sample(seed=subk1)
    # mu_post = tfd.Normal(post_mean, np.sqrt(var_post /post_lam)).sample(seed=subk2)
    # return np.array([mu_post, var_post]), key

@jit
def norm_lpdf(data, atoms):
    means = np.vstack([x[0] for x in atoms])
    sds = np.sqrt(np.vstack([x[1] for x in atoms]))
    
    return tfd.Normal(means[:, np.newaxis], sds[:, np.newaxis]).log_prob(data).T


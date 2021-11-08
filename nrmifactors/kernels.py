from nrmifactors.priors import NNIGPrior
import jax.numpy as np
from jax import jit, random
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from nrmifactors.priors import NNIGPrior

@jit
def nnig_update(alldata, clus_idx, c, mu0, lam, a, b, key):
    card = np.count_nonzero(clus_idx == c)
    ybar = np.nansum(np.where(clus_idx == c, alldata, 0)) / card
    var = np.nansum(np.where(clus_idx == c, (alldata - ybar)**2, 0)) / card

    post_mean = (lam * mu0 + ybar * card) / (lam + card)
    post_lam = lam + card
    post_shape = a + 0.5 * card
    post_rate = b + \
        0.5 * var * card + \
        0.5 * lam * card * (mu0 - ybar)**2 / post_lam
    key, subk1, subk2 = random.split(key, 3)
    var_post = tfd.InverseGamma(post_shape, post_rate).sample(seed=subk1)
    mu_post = tfd.Normal(post_mean, np.sqrt(var_post /post_lam)).sample(seed=subk2)
    return np.array([mu_post, var_post]), key

@jit
def norm_lpdf(data, atoms):
    means = np.vstack([x[0] for x in atoms])
    sds = np.sqrt(np.vstack([x[1] for x in atoms]))
    
    return tfd.Normal(means[:, np.newaxis], sds[:, np.newaxis]).log_prob(data).T


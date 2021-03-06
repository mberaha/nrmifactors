from nrmifactors.kernels import nnig_update, norm_lpdf
from nrmifactors.utils import run_hmc
from nrmifactors.priors import NrmiFacPrior
import jax.numpy as np
from jax import jit, lax, random, vmap
from jax.ops import segment_sum
from functools import partial
from sklearn.cluster import KMeans


from tensorflow_probability.substrates import jax as tfp
log_acceptance_proba_getter = \
    tfp.mcmc.simple_step_size_adaptation.hmc_like_log_accept_prob_getter_fn
tfd = tfp.distributions
tfb = tfp.bijectors

LOG_TARGET_ACCEPT_PROBA = np.log(0.75)


@partial(jit, static_argnums=(2,))
def update_gaussmix_atoms(data, clus, nclus, kp_mu0, kp_lam, kp_a, kp_b, rng_key):
    """
    Updates all the atoms of a Gaussian mixture model, conditional on the data
    and the cluster allocation.
    Each component is a Gaussian distribution with component-specific mean and 
    variance. The prior for the parameters is the Normal-Inverse-Gamma distribution,
    that is
        mean | var ~ N(kp_mu0, var / kp_lam)
               var ~ IG(kp_a, kp_b)  

    Parameters
    ----------
    data: an n_groups x ndata matrix (np.array) of floats. 
        If data has different cardinality in each  group, pad the matrix with np.nans
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix. The same padding with nans applies
    nclus: the number of components in the mixture model
    kp_mu0, kp_lam, kp_a, kp_b: the prior hyperparameters
    rng_key: the current rng state (instance of jax.random.PRNGKey) 
    """
    out = []
    for c in range(nclus):
        val, rng_key = nnig_update(
                data, clus, c, kp_mu0, kp_lam, kp_a, kp_b, rng_key)
        out.append(val)

    return np.vstack(out), rng_key


def update_Js_gamma(clus, lam, m, u, j_prior_a, j_prior_b, rng_key):
    """
    Updates the J parameters when J_k ~ Gamma(j_prior_a, j_prior_b), exploiting
    conjugacy

    Parameters
    ----------
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix
    lam: the num_groups x H Lambda matrix
    m: the H x K matrix of CoRM scores
    u: the g auxiliary variables
    j_prior_a, j_prior_b: the prior hyperparameters
    rng_key: the current rng state (instance of jax.random.PRNGKey) 
    """
    nclus = m.shape[1]
    out = []
    lam_m = np.matmul(lam, m)
    cards = np.array([np.sum(clus == c)
                     for c in np.arange(nclus)]).astype(float)
    a_post = cards + j_prior_a
    b_post = np.matmul(lam_m.T, u) + j_prior_b
    rng_key, subkey = random.split(rng_key)
    out = tfd.Gamma(a_post, b_post).sample(seed=subkey)
    return out, rng_key


def update_Js_beta(clus, lam, m, j, u, j_prior_a, j_prior_b, rng_key, 
                   step_size=0.0001, nsteps=50, niter=100):
    """
    Updates the J parameters when J_k ~ Beta(j_prior_a, j_prior_b), using
    Hamiltonian Monte Carlo

    Parameters
    ----------
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix
    lam: the num_groups x H Lambda matrix
    m: the H x K matrix of CoRM scores
    j: the current value of the J_k's
    u: the g auxiliary variables
    j_prior_a, j_prior_b: the prior hyperparameters
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """
    bijector = tfb.Logit()
    lm = np.matmul(lam, m)

    @jit
    def full_cond_lpdf(transformed_js, cluscount):
        prior = tfd.TransformedDistribution(
            tfd.Beta(
            j_prior_a, j_prior_b,
            force_probs_to_zero_outside_support=True),
            bijector=bijector)

        js = bijector.inverse(transformed_js)
        out = prior.log_prob(transformed_js)
        out -= np.sum(np.sum(lm * js, axis=1) * u)
        out += np.sum(np.log(js) * cluscount)
        return out

    cluscount = np.sum(clus[:, :, np.newaxis] == np.arange(j.shape[0]), axis=1)
    def target_lpdf(x): return full_cond_lpdf(x, cluscount)

    transformed_js = bijector.forward(j)

    rng_key, subkey = random.split(rng_key)
    nburn = niter - 2
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=2, num_burnin_steps=nburn, current_state=transformed_js,
        kernel=kernel, 
        trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.step_size, 
        seed=subkey)
    j = bijector.inverse(out[0][-1, :, :])
    step_size = out[1][-1]
    return j, rng_key, step_size


update_Js_jit = jit(update_Js_gamma)


def update_clus(data, lam, m, j, atoms, rng_key):
    """
    Updates the cluster allocations by sampling from their full conditional
    distribution, under a Gaussian mixture model

    Parameters
    ----------
    data: an n_groups x ndata matrix (np.array) of floats. 
        If data has different cardinality in each  group, pad the matrix with np.nans
    lam: the num_groups x H Lambda matrix
    m: the H x K matrix of CoRM scores
    atoms: the K component-specific mean and variances, stored as a K x 2 matrix
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """
    prior_probas = np.log(np.matmul(lam, m) * j)
    likes = norm_lpdf(data, atoms)
    probas = np.exp(prior_probas + likes)
    probas = np.exp(likes)
    probas /= np.nansum(probas, axis=1)[:, np.newaxis, :]
    rng_key, subkey = random.split(rng_key)
    clus = tfd.Categorical(logits=likes).sample(seed=subkey).T
    return clus, rng_key

update_clus_jit = jit(update_clus)

@jit
def update_lambda_unconstrained(
        clus, lam, m, j, u, lam_prior_a, lam_prior_b, rng_key, step_size=0.0001,
        nsteps=50, niter=100):
    """
    Update Lambda under the i.i.d. Gamma prior using HMC having transformed
    Lambda in the log-scale

    Parameters
    ----------
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix
    lam: the num_groups x H Lambda matrix
    m: the H x K matrix of CoRM scores
    j: the current value of the J_k's
    u: the g auxiliary variables
    lam_prior_b, lam_prior_b: the prior hyperparameters (shape and rate of Gamma)
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """
    bijector = tfb.Log()

    @jit
    def full_cond_lpdf(transformed_lam, cluscount, prior_shape, prior_rate):
        prior = tfd.TransformedDistribution(
            tfd.Gamma(
            prior_shape, prior_rate,
            force_probs_to_zero_outside_support=True),
            bijector=bijector)

        lam = bijector.inverse(transformed_lam)
        lm = np.matmul(lam, m)
        out = prior.log_prob(transformed_lam)
        out -= np.sum(np.sum(lm * j, axis=1) * u)
        out += np.sum(np.log(lm) * cluscount)
        return out

    cluscount = np.sum(clus[:, :, np.newaxis] == np.arange(j.shape[0]), axis=1)

    def target_lpdf(x): return full_cond_lpdf(x, cluscount, lam_prior_a, lam_prior_b)

    transformed_lam = bijector.forward(lam)

    rng_key, subkey = random.split(rng_key)
    nburn = niter - 2
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=2, num_burnin_steps=nburn, current_state=transformed_lam,
        kernel=kernel, 
        trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.step_size, 
        seed=subkey)
    lam = bijector.inverse(out[0][-1, :, :])
    step_size = out[1][-1]
    return lam, rng_key, step_size


@jit 
def lambda_like_lpdf(lam, m, j, u, cluscount):
    """
    Returns the evaluation of the log-Likelihood terms involving Lambda
    """
    lm = np.matmul(lam, m)
    out = - np.sum(np.sum(lm * j, axis=1) * u)
    out += np.sum(np.log(lm) * cluscount)
    return out

@jit
def update_lambda(
        clus, lam, m, j, u, lam_prior_a, lam_prior_b, rng_key, step_size=0.0001,
        nsteps=50, niter=100):
    """
    Update Lambda under the i.i.d. Gamma prior using HMC

    Parameters
    ----------
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix
    lam: the num_groups x H Lambda matrix
    m: the H x K matrix of CoRM scores
    j: the current value of the J_k's
    u: the g auxiliary variables
    lam_prior_b, lam_prior_b: the prior hyperparameters (shape and rate of Gamma)
    rng_key: the current rng state (instance of jax.random.PRNGKey).
    """

    @jit
    def prior_iid_gamma_lpdf(lam, prior_shape, prior_rate):
        out = np.sum(tfd.Gamma(
            prior_shape, prior_rate,
            force_probs_to_zero_outside_support=True).log_prob(lam))
        return out

    @jit
    def full_cond_lpdf(lam, cluscount, prior_shape, prior_rate):
        lm = np.matmul(lam, m)
        out = prior_iid_gamma_lpdf(lam, prior_shape, prior_rate) + \
               np.sum(np.log(lm) * cluscount) - np.sum(np.sum(lm * j, axis=1) * u)
        return out

    cluscount = np.sum(clus[:, :, np.newaxis] == np.arange(j.shape[0]), axis=1)

    def target_lpdf(x): return full_cond_lpdf(x, cluscount, lam_prior_a, lam_prior_b)

    rng_key, subkey = random.split(rng_key)
    nburn = niter - 2
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=2, num_burnin_steps=nburn, current_state=lam,
        kernel=kernel, 
        trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.step_size, 
        seed=subkey)
    lam = out[0][-1, :, :]
    step_size = out[1][-1]

    rng_key, subkey = random.split(rng_key)
    scale_factor = tfd.LogNormal(0., 0.5).sample(seed=subkey)
    prop_lam = lam * scale_factor
    arate = target_lpdf(prop_lam) - target_lpdf(lam)

    rng_key, subkey = random.split(rng_key)
    lam = lax.cond(arate > np.log(tfd.Uniform(0, 1).sample(seed=subkey)),
                   lambda _: prop_lam, lambda _: lam,
                   lam) 

    return lam, rng_key, step_size



def get_lambda_mgp(phis, deltas):
    """
    Transformation from the phi-delta parametrization of the MGP to
    the matrix Lambda
    """
    taus = np.cumprod(deltas, axis=-1)
    out = 1.0 / (phis * taus)
    return out


def update_lambda_mgp(
        clus, phis, deltas, m, j, u, mgp_nu, mgp_a1, mgp_a2, rng_key, step_size=0.0001,
        nsteps=50, niter=100):
    """
    Update Lambda (in the phi-delta parametrization) under the multiplicative 
    gamma process (MGP) prior prior using HMC.

    Parameters
    ----------
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix
    phis, deltas: the current values of the MGP
    m: the H x K matrix of CoRM scores
    j: the current value of the J_k's
    u: the g auxiliary variables
    mgp_nu, mgp_a1, mgp_a2: the prior hyperparameters
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """

    J = phis.shape[0]
    H = phis.shape[1]


    def phidelta_tovec(phis, deltas):
        return np.concatenate([phis.reshape(-1, ), deltas])


    def phidelta_fromvec(phidelta):
        phis = np.reshape(phidelta[:J*H], (J, H))
        deltas = phidelta[J*H:]
        return phis, deltas


    def mgp_prior(phis, deltas, nu, a1, a2):
        out = np.sum(tfd.Gamma(
            nu/2, nu/2,
            force_probs_to_zero_outside_support=True).log_prob(phis))
        out += tfd.Gamma(
            a1, 1,
            force_probs_to_zero_outside_support=True).log_prob(deltas[0])
        out += np.sum(tfd.Gamma(
            a2, 1,
            force_probs_to_zero_outside_support=True).log_prob(deltas[1:]))
        return out


    def full_cond_lpdf(phis_deltas, cluscount):
        phis, deltas = phidelta_fromvec(phis_deltas)
        lam = get_lambda_mgp(phis, deltas)
        lm = np.matmul(lam, m)
        out = mgp_prior(phis, deltas, mgp_nu, mgp_a1, mgp_a2) + \
              np.sum(np.log(lm) * cluscount) - np.sum(np.sum(lm * j, axis=1) * u)
        return out

    cluscount = np.sum(clus[:, :, np.newaxis] == np.arange(j.shape[0]), axis=1)

    def target_lpdf(x): return full_cond_lpdf(x, cluscount)

    rng_key, subkey = random.split(rng_key)
    nburn = niter - 2
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=2, num_burnin_steps=nburn, current_state=phidelta_tovec(phis, deltas),
        kernel=kernel, 
        trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.step_size, 
        seed=subkey)
    phis, deltas = phidelta_fromvec(out[0][-1, :])
    step_size = out[1][-1]

    return phis, deltas, rng_key, step_size


update_lambda_mgp_jit = jit(update_lambda_mgp)


def adapt_lambda_mgp(lam, m, phis, deltas, a1, a2, iter, 
                     mgp_nu, mgp_a1, mgp_a2, m_prior_a, m_prior_b,
                     threshold, rng_key):
    """
    Perform adaptation on the columns of Lambda under the MGP.
    If some columns of Lambda are "empty", they are removed. If none of them 
    are empty, a column is added by sampling it from the prior.
    The matrix M is modified accordingly
    """
    
    def non_empty_cols(lam, threshold):
        norms = np.sum(lam / np.sum(lam, axis=1)[:, np.newaxis], axis=0)
        return np.where(norms > threshold * np.mean(norms))[0]

    def remove_cols(args):
        lam, m, phis, deltas, rng_key = args
        keep_facs = non_empty_cols(lam, threshold)
        lam = lam[:, keep_facs]
        print("removing col, snew shape: ", lam.shape)
        phis = phis[:, keep_facs]
        deltas = deltas[keep_facs]
        m = m[keep_facs, :]
        return lam, m, phis, deltas, rng_key

    def add_col(args):
        lam, m, phis, deltas, rng_key = args
        print("adding column")
        rng_key, subkey = random.split(rng_key)
        deltas = np.concatenate(
            [deltas, tfd.Gamma(mgp_a2, 1).sample((1, ), seed=subkey)])
        rng_key, subkey = random.split(rng_key)
        new_phi_col = tfd.Gamma(mgp_nu/2, mgp_nu/2).sample(
            (phis.shape[0], 1), seed=subkey)
        phis = np.concatenate([phis, new_phi_col], axis=1)
        lam = get_lambda_mgp(phis, deltas)
        rng_key, subkey = random.split(rng_key)
        new_m_row = tfd.Gamma(m_prior_a, m_prior_b).sample(
            (1, m.shape[1]), seed=subkey)
        m = np.concatenate([m, new_m_row], axis=0)
        return lam, m, phis, deltas, rng_key

    def adapt(args):
        lam, m, phis, deltas, rng_key = args
        if len(non_empty_cols(lam, threshold)) < lam.shape[1]:
            return remove_cols(args)
        else:
            return add_col(args)

    rng_key, subkey = random.split(rng_key)
    lam, m, phis, deltas, rng_key = adapt((lam, m, phis, deltas, subkey))

    return lam, m, phis, deltas, rng_key


@partial(jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) COO sparse matrix
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    in_ = B.take(A.col, axis=0)
    prod = in_ * A.data[:, None]
    res = segment_sum(prod, A.row, shape)
    return res


@jit 
def multi_normal_prec_lpdf(x, mu, sigma, sigma_logdet, tau):
    """
    Log probability density function for a multivariate normal with precision
    matrix sigma, stored in a sparse matrix.
    """
    base = 0.5 * (np.log(tau) * sigma.shape[0])
    y = x - mu
    exp = - 0.5 * y.dot(sp_matmul(sigma, y.reshape(-1, 1), sigma.shape[0])) * tau
    return base + exp


@jit
def update_lambda_gmrf(
        clus, lam, m, j, u, sigma, sigma_logdet, tau, rng_key, 
        step_size=0.0001, nsteps=50, niter=100):
    """
    Update Lambda under the log-Gaussian Markov Random Field prior using HMC.

    Parameters
    ----------
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix
    lam: the current values of Lambda
    m: the H x K matrix of CoRM scores
    j: the current value of the J_k's
    u: the g auxiliary variables
    sigma: the prior precision stored in a sparse matrix
    sigma_logdet: log-determinant of the matrix sigma
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """

    bijector = tfb.Log()

    @jit
    def full_cond_lpdf(trans_lam):
        lam  = bijector.inverse(trans_lam)
        lm = np.matmul(lam, m)
        mu = - np.log(trans_lam.shape[1])
        prior = vmap(
            lambda x: multi_normal_prec_lpdf(
                x, mu, sigma, sigma_logdet, tau) +
            np.sum(bijector.inverse_log_det_jacobian(x))
            )(trans_lam.T)
        return np.sum(prior) + \
            np.sum(np.log(lm) * cluscount) - np.sum(np.sum(lm * j, axis=1) * u)
        

    cluscount = np.sum(clus[:, :, np.newaxis] == np.arange(j.shape[0]), axis=1)

    def target_lpdf(x): return full_cond_lpdf(x)

    rng_key, subkey = random.split(rng_key)
    nburn = niter - 2
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=2, num_burnin_steps=nburn, current_state=bijector.forward(lam),
        kernel=kernel, 
        trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.step_size, 
        seed=subkey)
    lam = bijector.inverse(out[0][-1, :, :])
    step_size = out[1][-1]

    rng_key, subkey = random.split(rng_key)
    scale_factor = tfd.LogNormal(0., 1.0).sample(seed=subkey)
    prop_lam = lam  * scale_factor
    arate = target_lpdf(bijector.forward(prop_lam)) - \
            target_lpdf(bijector.forward(lam))

    rng_key, subkey = random.split(rng_key)
    lam, accept = lax.cond(arate > np.log(tfd.Uniform(0, 1).sample(seed=subkey)),
                   lambda _: (prop_lam, 1), lambda _: (lam, 0),
                   lam) 

    return lam, accept, rng_key, step_size


@jit 
def update_tau_gmrf(tau, a, b, lam, sigma, sigma_logdet, rng_key, step_size=0.0001):
    """
    Update the tau parameter in the log-Gaussian Markov Random Field prior

    WARNING: this move produces poorly mixing chains due to the non-identifiability.
    We suggest to fix tau once and for all.
    Alternatively, very concentrated prior should be used.
    """
    bijector = tfb.Log()

    @jit
    def target_lpdf(log_tau):
        tau = bijector.inverse(log_tau)
        mu = -  np.log(lam.shape[1])
        out = np.sum(vmap(
            lambda x: multi_normal_prec_lpdf(
                x, mu, sigma, sigma_logdet, tau)
            )(np.log(lam).T))
        
        out += tfd.InverseGamma(a, b).log_prob(tau) + \
            bijector.inverse_log_det_jacobian(log_tau)
        return out

    rng_key, subkey = random.split(rng_key)
    kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target_lpdf)
    out = tfp.mcmc.sample_chain(
        num_results=2, num_burnin_steps=8, current_state=bijector.forward(tau),
        kernel=kernel, 
        trace_fn=None, 
        seed=subkey)
    log_tau = out[0]
    return bijector.inverse(log_tau), rng_key, step_size



def update_m(
        clus, lam, m, j, u, m_prior_a, m_prior_b, rng_key, step_size=0.0001,
        nsteps=50, niter=100):
    """
    Update M under the i.i.d. Gamma prior using HMC.

    Parameters
    ----------
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix
    lam: the current values of Lambda
    m: the H x K matrix of CoRM scores
    j: the current value of the J_k's
    u: the g auxiliary variables
    m_prior_a, m_prior_b: the hyperparameters of the prior
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """

    def full_cond_lpdf(m, cluscount, prior_shape, prior_rate):
        lm = np.matmul(lam, m)
        out = np.sum(tfd.Gamma(
            prior_shape, prior_rate,
            force_probs_to_zero_outside_support=True).log_prob(m))
        out -= np.sum(np.sum(lm * j, axis=1) * u)
        out += np.sum(np.log(lm) * cluscount)
        return out

    cluscount = np.sum(clus[:, :, np.newaxis] == np.arange(j.shape[0]), axis=1)
    def target_lpdf(x): return full_cond_lpdf(x, cluscount, m_prior_a, m_prior_b)

    rng_key, subkey = random.split(rng_key)
    nburn = niter - 1
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=2, num_burnin_steps=nburn, current_state=m,
        kernel=kernel, 
        trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.step_size, 
        seed=subkey)
    m = out[0][-1, :, :]
    step_size = out[1][-1]

    rng_key, subkey = random.split(rng_key)
    scale_factor = tfd.LogNormal(0., 0.5).sample(seed=subkey)
    prop_m = m * scale_factor
    arate = target_lpdf(prop_m) - target_lpdf(m)

    rng_key, subkey = random.split(rng_key)
    m = lax.cond(arate > np.log(tfd.Uniform(0, 1).sample(seed=subkey)),
                   lambda _: prop_m, lambda _: m,
                   m) 

    return m, rng_key, step_size


update_m_jit = jit(update_m)


def update_u(data, n, lam, m, j, rng_key):
    """
    Update the u_j's sampling from their full conditional distribution
    """
    t = np.sum(np.matmul(lam, m) * j, axis=1)
    rng_key, subkey = random.split(rng_key)
    out = tfd.Gamma(n, t).sample(seed=subkey)
    return out, rng_key

update_u_jit = jit(update_u)


@partial(jit, static_argnums=(0, 1, 2))
def update_lambda_and_m(
        ngroups, nlat, natoms, clus, lam, m, j, u, lam_prior_a, lam_prior_b,  
        m_prior_a, m_prior_b,
        rng_key, step_size=0.0001, nsteps=50, niter=100):
    """
    Joint update of Lambda and M from the joint full conditional, using HMC,
    when Lambda has i.i.d. entries from a Gamma distribution (generalizations
    to other priors for Lambda are straightforward).
    This move has usually a low-acceptance rate given the dimensionality of
    the space, but might result in better exploration of the posterior, especially
    in the first iterations.

    Parameters
    ----------
    ngroups: the number of groups in the data
    nlat: the number of latent measures
    natoms: the number of support points
    clus: an n_groups x ndata matrix (np.array) of integers.
        Each entry is the cluster allocation for the corresponding element in the
        data matrix
    lam: the current values of Lambda
    m: the H x K matrix of CoRM scores
    j: the current value of the J_k's
    u: the g auxiliary variables
    lam_prior_a, lam_prior_b, m_prior_a, m_prior_b: the prior hyperparameters
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """

    bijector = tfb.Log()
    
    @jit
    def sperate_lam_m(lam_and_m):
        lam = lam_and_m[:(ngroups * nlat)].reshape(ngroups, nlat)
        m = lam_and_m[(ngroups * nlat):].reshape(nlat, natoms)
        return lam, m

    @jit
    def concatenate_lam_m(lam, m):
        return np.concatenate([lam.reshape(-1,), m.reshape(-1, )])


    @jit
    def full_cond_lpdf(log_lam_and_m):
        prior_lam = tfd.TransformedDistribution(
            tfd.Gamma(
            lam_prior_a, lam_prior_b,
            force_probs_to_zero_outside_support=True),
            bijector=bijector)

        prior_m = tfd.TransformedDistribution(
            tfd.Gamma(
            m_prior_a, m_prior_b,
            force_probs_to_zero_outside_support=True),
            bijector=bijector)

        log_lam, log_m = sperate_lam_m(log_lam_and_m)
        lam = bijector.inverse(log_lam)
        m = bijector.inverse(log_m)
        lm = np.matmul(lam, m)
        out = np.sum(prior_lam.log_prob(log_lam)) + np.sum(prior_m.log_prob(log_m))
        out -= np.sum(np.sum(lm * j, axis=1) * u)
        out += np.sum(np.log(lm) * cluscount)
        return out

    cluscount = np.sum(clus[:, :, np.newaxis] == np.arange(j.shape[0]), axis=1)

    def target_lpdf(x): return full_cond_lpdf(x)

    lam_and_m = concatenate_lam_m(lam, m)
    unconstrained_lam_and_m = bijector.forward(lam_and_m)

    rng_key, subkey = random.split(rng_key)
    nburn = niter - 2
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=2, num_burnin_steps=nburn, current_state=unconstrained_lam_and_m,
        kernel=kernel, 
        trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.step_size, 
        seed=subkey)
    lam_and_m = bijector.inverse(out[0][-1, :])
    lam, m = sperate_lam_m(lam_and_m) 
    step_size = out[1][-1]
    return lam, m, rng_key, step_size


def run_one_step(state, data, nan_idx, nobs_by_group, prior, rng_key):
    """
    Runs one step of the MCMC algorithm as in the paper

    Parameters
    ----------
    state: instance of state.State. The current state of the MCMC
    data: an n_groups x ndata matrix (np.array) of floats. 
        If data has different cardinality in each  group, pad the matrix with np.nans
    nan_idx: an n_groups x ndata matrix (np.array) of bools. 
        Each entry is equal to True if the corresponding observation is a nan
    nobs_by_groups: the number of observations in each group
    prior: an instance of priors.NrmiFacPrior
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """

    nclus = state.atoms.shape[0]
    state.clus, rng_key = update_clus_jit(
        data, state.lam, state.m, state.j, state.atoms, rng_key)
    state.clus = state.clus.at[nan_idx].set(-10)

    # state.lam, state.m, rng_key, state.lam_step_size = update_lambda_and_m(
    #         data.shape[0], state.m.shape[0], state.atoms.shape[0], 
    #         state.clus, state.lam, state.m, state.j, state.u, 
    #         prior.lam_prior_iid.a, prior.lam_prior_iid.b,
    #         prior.m_prior.a, prior.m_prior.b,
    #         rng_key, step_size=state.lam_step_size)

    state.atoms, rng_key = update_gaussmix_atoms(
        data, state.clus, nclus, prior.kern_prior.mu0, prior.kern_prior.lam, prior.
        kern_prior.a, prior.kern_prior.b, rng_key)


    state.j, rng_key = update_Js_jit(
        state.clus, state.lam, state.m, state.u, prior.j_prior.a, 
        prior.j_prior.b, rng_key)

    if prior.lam_prior == "mgp":
        state.phis, state.deltas, rng_key, state.lam_step_size = update_lambda_mgp_jit(
            state.clus, state.phis, state.deltas, state.m, state.j, state.u, 
            prior.lam_prior_mgp.nu, prior.lam_prior_mgp.a1, prior.lam_prior_mgp.a2,
            rng_key, step_size=state.lam_step_size)
        
        state.lam = get_lambda_mgp(state.phis, state.deltas)

    elif prior.lam_prior == "gmrf":
        state.lam, state.accept_lam_scaling, rng_key, state.lam_step_size = update_lambda_gmrf(
            state.clus, state.lam, state.m, state.j, state.u, 
            prior.lam_prior_gmrf.sigma, 
            prior.lam_prior_gmrf.sigma_logdet, 
            state.tau, rng_key, 
            step_size=state.lam_step_size)

    else:
        state.lam, state.m, rng_key, state.lam_step_size = update_lambda_and_m(
            data.shape[0], state.m.shape[0], state.atoms.shape[0], 
            state.clus, state.lam, state.m, state.j, state.u, 
            prior.lam_prior_iid.a, prior.lam_prior_iid.b,
            prior.m_prior.a, prior.m_prior.b,
            rng_key, step_size=state.lam_step_size)

    
    state.m, rng_key, state.m_step_size = update_m_jit(
        state.clus, state.lam, state.m, state.j, state.u, 
        prior.m_prior.a, prior.m_prior.b, rng_key, step_size=state.m_step_size) 

    state.u, rng_key = update_u_jit(
        data, nobs_by_group, state.lam, state.m, state.j, rng_key)

    state.iter += 1

    return state, rng_key



def adapt_mgp(init_state, niter, adapt_every, data, nan_idx, nobs_by_group, prior, rng_key):
    """
    Runs the adaptation phase of the MCMC algorithm. Used only when Lambda has
    a multiplicative gamma process prior.
    This function samples "niter" times from the full conditionals, every
    "adapt_every" iterations, the "adapt_lambda_mgp" is called, and 
    all the functions are re-compiled (jit).
    
    The time required to run each iteration quickly decreases with time, so
    don't be scared if the first iterations take very long.
    Moreover, once the number of columns of Lambda settles around 2/3 values,
    compilation is cached and the runtime decreases dramatically

    Parameters
    ----------
    init_state: instance of state.State. The initial state of the MCMC
    niter: number of iterations of the adaptation phase 
    adapt_every: number of iterations between every attempt to adapt
    data: an n_groups x ndata matrix (np.array) of floats. 
        If data has different cardinality in each  group, pad the matrix with np.nans
    nan_idx: an n_groups x ndata matrix (np.array) of bools. 
        Each entry is equal to True if the corresponding observation is a nan
    nobs_by_groups: the number of observations in each group
    prior: an instance of priors.NrmiFacPrior
    rng_key: the current rng state (instance of jax.random.PRNGKey)
    """
    state = init_state
    nclus = state.atoms.shape[0]

    update_Js_jit = jit(update_Js_gamma)
    update_clus_jit = jit(update_clus)
    update_lambda_mgp_jit = jit(update_lambda_mgp)
    get_lambda_mgp_jit = jit(get_lambda_mgp)
    update_m_jit = jit(update_m)
    update_u_jit = jit(update_u)
    
    for i in range(niter):
        print("\r{0}/{1}".format(i+1, niter), flush=True, end=" ")
        state.clus, rng_key = update_clus_jit(
            data, state.lam, state.m, state.j, state.atoms, rng_key)
        state.clus = state.clus.at[nan_idx].set(-10)

        state.atoms, rng_key = update_gaussmix_atoms(
            data, state.clus, nclus, prior.kern_prior.mu0, prior.kern_prior.lam, prior.
            kern_prior.a, prior.kern_prior.b, rng_key)

        state.j, rng_key = update_Js_jit(
            state.clus, state.lam, state.m, state.u, prior.j_prior.a, 
            prior.j_prior.b, rng_key)

        state.phis, state.deltas, rng_key, state.lam_step_size = update_lambda_mgp_jit(
            state.clus, state.phis, state.deltas, state.m, state.j, state.u, 
            prior.lam_prior_mgp.nu, prior.lam_prior_mgp.a1, prior.lam_prior_mgp.a2,
            rng_key, step_size=state.lam_step_size)
        
        state.lam = get_lambda_mgp_jit(state.phis, state.deltas)
        
        state.m, rng_key, state.m_step_size = update_m_jit(
            state.clus, state.lam, state.m, state.j, state.u, 
            prior.m_prior.a, prior.m_prior.b, rng_key, step_size=state.m_step_size) 

        state.u, rng_key = update_u_jit(
            data, nobs_by_group, state.lam, state.m, state.j, rng_key)

        if (i+1) % adapt_every == 0:
            state.lam, state.m, state.phis, state.deltas, rng_key = adapt_lambda_mgp(
                state.lam, state.m, state.phis, state.deltas, prior.lam_prior_mgp.adapt_a1,
                prior.lam_prior_mgp.adapt_a2, state.iter, prior.lam_prior_mgp.nu,
                prior.lam_prior_mgp.a1, prior.lam_prior_mgp.a2, 
                prior.m_prior.a, prior.m_prior.b,
                prior.lam_prior_mgp.adapt_threshold, rng_key)
            
            update_Js_jit = jit(update_Js_gamma)
            update_clus_jit = jit(update_clus)
            update_lambda_mgp_jit = jit(update_lambda_mgp)
            update_m_jit = jit(update_m)
            update_u_jit = jit(update_u)

    state.iter += 1

    return state, rng_key

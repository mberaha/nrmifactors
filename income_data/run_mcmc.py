import os
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["xla_cpu_multi_thread_eigen"] = "True"
os.environ["intra_op_parallelism_threads"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"


import jax.numpy as np
from jax import random, vmap
from jax.experimental.sparse import COO

from nrmifactors import algorithm as algo
from nrmifactors.state import State
import nrmifactors.priors as priors
import pickle
from sklearn.cluster import KMeans

from copy import deepcopy

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


import logging
logger = logging.getLogger("root")

class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger.addFilter(CheckTypesFilter())


def eval_pred_dens(data, data2group, lam, m, j, atoms):
    weights = np.matmul(lam, m) * j
    weights /= weights.sum(axis=1)[:, np.newaxis]
    data_flat = data.reshape(-1, 1)
    data2group_flat = data2group.reshape(-1, 1)
    eval_comps = tfd.Normal(loc=atoms[:, 0], scale=np.sqrt(atoms[:, 1])).prob(
        data_flat[:, np.newaxis])
    dens = eval_comps[:, np.newaxis, :] * weights[np.newaxis, data2group_flat, :]
    dens = np.sum(dens, axis=-1).T
    return dens


def fit(prior, nlat, init_atoms, data):
    key = random.PRNGKey(0)

    niter = 20000
    nburn = 10000
    thin = 1

    ngroups = data.shape[0]
    nan_idx = np.where(np.isnan(data))
    nobs_by_group = np.array(
            [np.count_nonzero(~np.isnan(x)) for x in data]).astype(float)

    lam = np.ones((data.shape[0], nlat))
    m = tfd.Gamma(prior.m_prior.a, prior.m_prior.b).sample(
        (nlat, natoms), seed=key).astype(float)

    j = np.ones(natoms).astype(float) * 0.5
    u = np.ones(ngroups).astype(float)

    clus = tfd.Categorical(probs=np.ones(natoms)/natoms).sample(data.shape, seed=key)
    clus = clus.at[np.isnan(data)].set(-10)  
    state = State(iter=0, atoms=init_atoms, j=j, lam=lam, m=m, 
                  clus=clus, u=u, tau=2.5)

    states = []
    for i in range(niter):
        print("\r{0}/{1}".format(i+1, niter), flush=True, end=" ")
        state, key = algo.run_one_step(state, data, nan_idx, nobs_by_group, prior, key)
        if (i > nburn) and (i % thin == 0):
            states.append(deepcopy(state))

    # data2group = np.stack([np.ones(data.shape[1]) * i for i in range(data.shape[0])])
    # eval_densities = vmap(lambda x: eval_densities(data, data2group, *x))(
    #     [(x.lam, x.m, x.j, x.atoms) for x in states])

    with open("income_data/california_mcmc_out_lat{0}.pickle".format(nlat), "wb") as fp:
        pickle.dump({"states": states}, fp)


if __name__ == "__main__":
    with open("income_data/california_puma_neighbors.pickle", "rb") as fp:
        W = pickle.load(fp)
        W = np.array(W)

    with open("income_data/california_income_subsampled.pickle", "rb") as fp:
        data = pickle.load(fp)
        data = np.array(data)
        data = data.at[data < 0].set(np.nan)
        data = np.log(data)
        print("data.shape: ", data.shape)
    
    prec = np.diag(W.sum(axis=1)) - 0.95 * W
    eigvals, eigvecs = np.linalg.eigh(prec)
    prec_logdet = np.sum(np.log(eigvals[eigvals > 1e-6]))
    prec = COO.fromdense(prec)

    natoms = 20
    km = KMeans(natoms)
    km.fit(data.reshape(-1, 1)[~np.isnan(data.reshape(-1, 1))].reshape(-1, 1))
    means = km.cluster_centers_
    init_atoms = np.hstack([means, np.ones_like(means) * 0.3])
    prior = priors.NrmiFacPrior(
        kern_prior=priors.NNIGPrior(0.0, 0.01, 5.0, 5.0),
        lam_prior_gmrf=priors.GMRFPrior(sigma=prec, sigma_logdet=prec_logdet,
                                        tau_a=2, tau_b=2),
        lam_prior="gmrf",
        m_prior=priors.GammaPrior(2.0, 2.0),
        j_prior=priors.GammaPrior(2.0, 2.0))

    for nlat in [8, 10]:
        fit(prior, nlat, init_atoms, data)
    
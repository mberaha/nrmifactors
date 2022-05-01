import argparse
import os
os.environ["JAX_ENABLE_X64"] = "true"

import pickle
import jax.numpy as np

from copy import deepcopy
from jax import random
from sklearn.cluster import KMeans

from nrmifactors import algorithm as algo
from nrmifactors.state import State
import nrmifactors.priors as priors

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


import logging
logger = logging.getLogger("root")

class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger.addFilter(CheckTypesFilter())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--chains_file", type=str)
    args = parser.parse_args()

    seed = 0
    key = random.PRNGKey(seed)

    with open(args.data_file, "rb") as fp:
        data = pickle.load(fp)
    
    data = np.array(data)
    ngroups = data.shape[0]
    nan_idx = np.where(np.isnan(data))
    nobs_by_group = np.array(
            [np.count_nonzero(~np.isnan(x)) for x in data]).astype(float)

    natoms = 10
    nlat = 5 ## adaptation gives 5 latent measures

    # initialize stuff
    km = KMeans(natoms)
    km.fit(data.reshape(-1, 1)[~np.isnan(data.reshape(-1, 1))].reshape(-1, 1))
    means = km.cluster_centers_
    init_atoms = np.hstack([means, np.ones_like(means) * 0.3])

    prior = priors.NrmiFacPrior(
        kern_prior=priors.NNIGPrior(0.0, 0.01, 5.0, 5.0),
        lam_prior_iid=priors.GammaPrior(4.0, 4.0),
        lam_prior_mgp=priors.MGPPrior(5.0, 2.0, 3.5, 0, -0.05, 0.025),
        lam_prior="mgp",
        m_prior=priors.GammaPrior(2.0, 2.0),
        j_prior=priors.GammaPrior(2.0, 2.0)
    )


    lam = np.ones((data.shape[0], nlat))
    m = tfd.Gamma(prior.m_prior.a, prior.m_prior.b).sample(
        (nlat, natoms), seed=key).astype(float)

    j = np.ones(natoms).astype(float) * 0.5
    u = np.ones(ngroups).astype(float)

    clus = tfd.Categorical(probs=np.ones(natoms)/natoms).sample(data.shape, seed=key)
    clus = clus.at[np.isnan(data)].set(-10) 
    state = State(
        iter=0,
        atoms=init_atoms, 
        j=j, 
        lam=lam,
        phis=1.0/lam,
        deltas=np.ones(lam.shape[1]),
        m=m, 
        clus=clus, 
        u=u,
    )
    niter = 11000
    nburn = 10000
    thin = 1

    # state, key = algo.adapt_mgp(state, 1000, 50, data, nan_idx, nobs_by_group, prior, key)

    states = []

    for i in range(niter):
        print("\r{0}/{1}".format(i+1, niter), flush=True, end=" ")
        state, key = algo.run_one_step(state, data, nan_idx, nobs_by_group, prior, key)
        if (i > nburn) and (i % thin == 0):
            states.append(deepcopy(state))

    with open(args.chains_file, "wb") as fp:
        pickle.dump(states, fp)

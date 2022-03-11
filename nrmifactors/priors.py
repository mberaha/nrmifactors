import jax.numpy as np
from jax.experimental.sparse import COO
from dataclasses import dataclass
from typing import Optional


@dataclass
class NNIGPrior:
    mu0: float
    lam: float
    a: float
    b: float

@dataclass
class GammaPrior:
    a: float
    b: float

@dataclass
class MGPPrior:
    nu: float
    a1: float
    a2: float

@dataclass
class GMRFPrior:
    sigma: COO
    sigma_logdet: float
    tau_a: float
    tau_b: float

@dataclass
class NrmiFacPrior:
    kern_prior: NNIGPrior
    lam_prior: str
    m_prior: GammaPrior
    j_prior: GammaPrior
    lam_prior_iid: Optional[GammaPrior] = None
    lam_prior_mgp: Optional[MGPPrior] = None
    lam_prior_gmrf: Optional[GMRFPrior] = None




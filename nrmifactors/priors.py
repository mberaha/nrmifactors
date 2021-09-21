import jax.numpy as np
from dataclasses import dataclass


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
class NrmiFacPrior:
    kern_prior: NNIGPrior
    lam_prior: GammaPrior
    m_prior: GammaPrior
    j_prior: GammaPrior




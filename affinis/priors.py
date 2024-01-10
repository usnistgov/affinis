#!/usr/bin/env python3
from plum import dispatch
from typing import Callable
import numpy as np
# @dispatch
def beta_binom_post(a:float=0.5, b:float=0.5, succ:int, n:int):
    return (succ+a)/(a+b+n)

@dispatch
def pseudocts(pseudocts:float, numer: np.NDArray, denom:np.NDArray):
    return (numer+pseudocts)/(denom+2*pseudocts)

@dispatch
def pseudocts(prior:tuple[float, float], numer: np.NDArray, denom: np.NDArray):
    a,b = prior
    return (a+numer)/(a+b+denom)

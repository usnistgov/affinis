from __future__ import annotations
from plum import dispatch
# from numbers import Number
from typing import Callable, Literal, TypeAlias
import numpy as np
from jaxtyping import Num
from affinis.utils import _sq

Number: TypeAlias = float | int
ElemWise: TypeAlias = Num[np.ndarray, "*elems"]
ElemReduceFunc: TypeAlias = Callable[[ElemWise, ElemWise], ElemWise]

PsdCtOpts:TypeAlias = Literal["min-connect","zero-sum"]
PsdCts: TypeAlias = tuple[Number, Number] | tuple[PsdCtOpts, Number] | PsdCtOpts | Number

def _safe_div(num: ElemWise, den: ElemWise) -> ElemWise:
    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=den != 0,
    )


@dispatch.abstract
def pseudocount(
    prior: PsdCts
) -> ElemReduceFunc:
    """Additive binomial smoothing via beta prior (beta-binomial)"""
    ...


@dispatch
def pseudocount(prior: Number) -> ElemReduceFunc:
    """additiv smoothing binomial with symmetric beta prior (a = b) 
    
    Common cases for a:
    
    - Haldane: \t 0.
    - Laplace: \t 1.
    - Jeffreys: \t 0.5

    Args:
      prior:  

    Returns: callable combining numerator and denominator 

    """    
    def _beta_binom_post(num: ElemWise, den: ElemWise) -> ElemWise:
        return _safe_div(
            num + prior,
            den + 2 * prior,
        )

    return _beta_binom_post


@dispatch
def pseudocount(prior: tuple[Number, Number]) -> ElemReduceFunc:
    """additive smoothing binomial with (possibly) asymmetric prior

    Args:
      prior: alpha,beta parameters of beta prior 

    Returns: callable combining numerator and denominator 

    """
    a, b = prior

    def _beta_binom_post(num: ElemWise, den: ElemWise) -> ElemWise:
        return _safe_div(num + a, den + a + b)

    return _beta_binom_post


@dispatch(precedence=1)
def pseudocount(prior: Literal["min-connect"]) -> ElemReduceFunc:
    r"""additive smoothing binomial with asymmetric prior biasing sparsity
    
    if observations are trials over an array of graph edges.
    
    The number of edges that are on or off in a graph is "zero sum"... One extra "on" means
    one less "off.
    So, the proportion of time we will be observing an "on" edge might be thought of as a Wiener
    Process, and thus follows a (generalized) arcsine distribution.
    This means we need a "bathtub" prior a+b=1.
    
    For a concave beta (a,b < 1), the anti-mode is the least likely spot, with the two
    (limiting) modes being at 0,1.
    If a complete graph has n(n-2)/2 edges, while a min. connected one has n-1, then
    we can bias toward non-edges such that the least-likely p is the ratio 
    
    .. math::
        \frac{1-(n-1)}{\frac{n*(n-1)}{2}}
    
    This comes out to a=2/n, b=1-2/n, so 

    .. math:: 
        P(p|a,b) = \frac{\textrm{\#successes}+2/n}{\textrm{\#trials}+1}

    Args:
      prior: "min-connect" 



    """
    def _beta_binom_post(num: ElemWise, den: ElemWise) -> ElemWise:
        n = _sq(num).shape[0]
        # n_nodes = num.shape[1]
        # n_pairs = n_nodes * (n_nodes - 1) / 2.0
        return _safe_div(num + 2.0 / n, den + 1.0)

    return _beta_binom_post


@dispatch
def pseudocount(prior: tuple[Literal["zero-sum"], Number]) -> ElemReduceFunc:
    """TODO derive the approx-cts for projection onto simplex

    unlike the other methods, this directly returns the `a` values for a      
    Beta(a, 1-a) distribution. For use when the full
    Beta distribution is desired e.g. for active learning or uncertainty.

    (it turns out that a/(a+1-a) == a, so this is also the mean of the distribution)
    """
    a = prior[1]
    # b = 1 - a

    def _beta_binom_post(suc: ElemWise, tot: ElemWise) -> ElemWise:
        c = (suc - a * tot) / (tot + 1)
        a_n = a + c
        # b_n = 1 - a_n
        return a_n
        # return _safe_div(a_n, a_n + b_n)

    return _beta_binom_post


@dispatch
def pseudocount(
    prior: tuple[Literal["zero-sum"], Literal["min-connect"]]
) -> ElemReduceFunc:
    """A combination of a zero-sum (a, 1-a) beta prior and a=2/n for tree-like"""

    def _beta_binom_post(suc: ElemWise, tot: ElemWise) -> ElemWise:
        n = _sq(suc).shape[0]
        a = 2 / n
        # b = 1 - a
        c = (suc - a * tot) / (tot + 1)
        a_n = a + c
        # b_n = 1 - a_n
        # return _safe_div(a_n, a_n + b_n)
        return a_n

    return _beta_binom_post

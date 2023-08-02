import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from .utils import _sq, _outer

"""
All functions in this module take in data as _design matrices_ 
(i.e. observations x features), and return a feature association measure 
(i.e. features x features). 

Note that some of these functions return valid adjacency matrices (e.g. a feature 
is not associated to itself), while others return covariance or correlations (features 
are partially or fully correlated to themselves). 

Where appropriate, the methods here allow for additive/laplace smoothing, even in cases 
where this is not traditionally done (like cosine similarity). We give interpretations
of meaning that allow for this, where we can.

TODO: define type signatures by array shapes
TODO: dispatch on dataframes, static-frames, etc.
"""

def coocur_prob(X, pseudocts=0.0):
    """probability of a co-ocurrence per-observation"""
    cts = X.T @ X + pseudocts
    tot = X.shape[0] + 2 * pseudocts
    return cts / tot


# def cond_prob(X, pseudocts=0.):
#     cts = X.T@X + pseudocts
#     tot = X.sum(axis=0)+2
#     return cts/tot


def _marginal_cts(X):
    return X.sum(axis=0), (1 - X).sum(axis=0)


def _marginal_prob(X):
    return [ct / X.shape[0] for ct in _marginal_cts(X)]


def _contingency_cts(X):
    both = _sq(X.T @ X)
    one = _sq(X.T @ (1 - X))
    other = _sq((1 - X).T @ X)
    neither = _sq((1 - X).T @ (1 - X))
    return neither, one, other, both


def _contingency_prob(X, pseudocts=0.0):
    cts = np.vstack(_contingency_cts(X)) + pseudocts
    return [ct for ct in cts / (X.shape[0] + pseudocts * 2)]


def _binary_contingency(X):
    n = X.shape[0]
    both = coocur_prob(X)  
    neither = coocur_prob(1 - X)  
    one = (X.T @ (1 - X) + 1) / (
        n + 2
    )  # np.logical_and.outer(X, ~X).mean(axis=2).mean(axis=0)
    other = one.T
    return neither, one, other, both


def odds_ratio(X, pseudocts=0.0):
    """Ratio of the odds of a true pos/neg to false pos/neg
    
    For associations, we replace pos/neg and true/false with 
    a=yes/no and b=yes/no.  
    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(a * d / (b * c)) + np.eye(X.shape[1])


def MI_binary(X, pseudocts=0.0):
    """Mutual Information over binary random variables
    
    For use in e.g. Chow-Liu Trees
    """
    Pxy = np.array(_contingency_prob(X, pseudocts=pseudocts))

    # Pxy = np.vstack([sq(i) for i in cond_table])
    yes, no = _marginal_prob(X)

    # print(Pxy.sum(axis=1))
    PxPy = np.vstack(
        [
            _sq(np.multiply.outer(a, b))
            for a, b in [(no, no), (yes, no), (no, yes), (yes, yes)]
        ]
    )
    entropy = -np.vstack([x * np.log(x) for x in (yes, no)]).sum(axis=0)

    MI_pairs = (Pxy * np.log(Pxy) - Pxy * np.log(PxPy)).sum(axis=0)
    return _sq(MI_pairs) + np.diag(entropy)
    # return sq(entropy(Pxy, PxPy))


def chow_liu(X, pseudocts=0.0):
    """Chow-Liu Tree on the features of a (binary) design matrix
    
    computes mutual information over all pairs of features, and returns
    the maximum spanning tree on them. Assumes a symmetric adjacency is wanted. 
    """
    # return sq(sq(minimum_spanning_tree(-MI_binary(X)).todense()))
    return _sq(
        _sq(
            minimum_spanning_tree(
                np.exp(-(MI_binary(X, pseudocts=pseudocts)))
            ).todense()
        )
    )


def yule_y(X, pseudocts=0.0):
    """a.k.a. Coefficient of Colligation. 
    
    mobius transform of the Odds Ratio to the range [-1,1]
    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(((ad := np.sqrt(a * d)) - (bc := np.sqrt(b * c))) / (ad + bc))
    # return ((sor:=np.sqrt(OR))-1)/(sor+1)
    # return ((ad:=np.sqrt(both*neither))-(bc:=np.sqrt(one*other)))/(ad+bc)


def yule_q(X, pseudocts=0.0):
    """a.k.a. Goodman & Kruskal's gamma for 2x2. 

    mobius transform of the Odds Ratio to the range [-1,1]
    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(((ad := a * d) - (bc := b * c)) / (ad + bc))


def ochiai(X, pseudocts=0.0):
    """AKA cosine similarity on binary sets

    This code illustrates the idea that we can interpret it as conditional probability:
    The "exposure" of pairwise co-occurrences can't be larger than the sample size,
    so instead we approximate it as a psuedo-variable  having the geometric average of
    two original exposure rates $\sqrt{X_1X_2}$

    This interpretation has a nice side-effect of letting us "smooth" the measure with
    laplace/additive pseudocounts on each bernoulli(-ish) "co-occurrence variable".
    """
    # I = np.eye(X.shape[-1])
    co_occurs = _sq(X.T @ X) + pseudocts
    exposures = X.sum(axis=0)
    pseudo_exposure = np.sqrt(_outer(np.multiply, exposures)) + 2 * pseudocts
    return _sq(co_occurs) / pseudo_exposure + np.eye(X.shape[1])

def binary_cosine_similarity(X, pseudocts=0.):
    """alias of `ochiai(X), provided for user convenience"""
    return ochiai(X, pseudocts=pseudocts)


def resource_project(X):
    """bipartite project due to Zhao et al.

    Really just a step of sinkhorn-knopp on the bipartite adjacency
    https://doc.rero.ch/record/8503/files/zhang_bnp.pdf
    """
    P = ((X.T) / (X.sum(axis=1))) @ ((X) / (X.sum(axis=0)))
    return np.maximum(P, P.T)

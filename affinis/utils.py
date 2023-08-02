from functools import cache
from typing import Callable

import numpy as np
from bidict import frozenbidict
from scipy.linalg import lapack
from scipy.sparse import coo_array
from scipy.spatial.distance import squareform


def _outer(f:Callable[[np.ndarray,np.ndarray],np.ndarray], a:np.ndarray):
    """do a thing to every combination of entries in an array"""
    return f.outer(a,a)

def _sq(A):
    """we want the off-diagonals, flat<->sq
    
    Typing this out got old. 
    """
    return squareform(A, checks=False)

def minmax(x, axis=None): 
    return (x-x.min(axis=axis))/(x.max(axis=axis)-x.min(axis=axis))

def norm_diag(A):
    a_ii = np.diag(A)
    
    return A/_outer(np.multiply, np.sqrt(a_ii))


def _diag(A):
    return np.diag(np.diag(A))


@cache
def _tri_inds_cache(n):
    return np.tri(n, k=-1, dtype=bool)


@cache
def _std_vec(n: int, i: int):
    return np.eye(n)[i]


@cache
def _map_edge_to_nodes(n: int):
    return frozenbidict(enumerate(zip(*[list(i) for i in np.triu_indices(n, k=1)])))


def _e_to_ij(n: int, e: int) -> tuple[int, int]:
    return _map_edge_to_nodes(n)[e]


def _ij_to_e(n: int, ij: tuple[int, int]):
    return _map_edge_to_nodes(n).inverse[ij]


def _upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    inds = _tri_inds_cache(n)
    ut[inds] = ut.T[inds]


def _fast_PD_inverse(m):
    """using `lapack.dpotrf` with cached indexing for PosDef matrices.
    Thanks to [Kerrick Staley](https://stackoverflow.com/a/58719188)
    #TODO re-write with safer/better errors (coconut? beartype? plum?)
    """
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError("dpotrf failed on input {}".format(m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError("dpotri failed on input {}".format(cholesky))
    _upper_triangular_to_symmetric(inv)
    return inv


def _rank1_downdate(B, u):
    """!!INPLACE!! no cholesky downdating in scipy yet... julia tho?"""
    u = np.atleast_2d(u)
    B += (B @ u.T) @ (u @ B) / (1 - u @ B @ u.T)


def _rank1_update(B, u):
    """!!INPLACE!! #TODO replace with chebotarev/avrachenkov update rule for forests?"""
    u = np.atleast_2d(u)
    B -= (B @ u.T) @ (u @ B) / (1 + u @ B @ u.T)


@cache
def _std_incidence_vec(n: int, ij: tuple[int, int]):
    """oriented incidence vector for n nodes and (source,target) edge tuple"""
    return _std_vec(n, ij[1]) - _std_vec(n, ij[0])


@cache
def _map_edge_to_stdvec(n: int):
    return frozenbidict(
        {e: tuple(_std_incidence_vec(n, ij)) for e, ij in _map_edge_to_nodes(n).items()}
    )


def sparse_adj_to_incidence(A):
    """turns (sparse, directed, assymmetric) adjacency array into incidence matrix.

    (-source,+target) oriented, such that the positive node is always the COO row index
    """
    S = coo_array(A)
    n = S.col.shape[0]
    ones, idx = np.ones(n), np.arange(n)
    shape = (n, S.shape[0])
    return coo_array((ones, (idx, S.row)), shape=shape) - coo_array(
        (ones, (idx, S.col)), shape=shape
    )

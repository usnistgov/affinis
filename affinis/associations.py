
from __future__ import annotations
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path, reconstruct_path
from scipy.sparse import coo_array, issparse, sparray
import sparse
from jaxtyping import Bool, Num
from typing import TypeAlias
from .utils import (
    _sq,
    _outer,
    _sparse_directed_to_symmetric,
    groupby_col0,
    complete_edgelist_on_nodes,
    sq_ij_e,
    edge_weights_to_laplacian,
    _norm_diag,
)
from .priors import pseudocount, PsdCts
from .distance import adjusted_forest_dists

__doc__ = """
All functions in this module take in data as design matrices
(i.e. observations x features), and return a feature association measure 
(i.e. features x features). 

Note that some of these functions return valid adjacency matrices (e.g. a feature 
is not associated to itself), while others return covariance or correlations (features 
are partially or fully correlated to themselves). 

Where appropriate, the methods here allow for additive/laplace smoothing, even in cases 
where this is not traditionally done (like cosine similarity). We give interpretations
of meaning that allow for this, where we can.
"""

# TODO: define type signatures by array shapes
# TODO: dispatch on dataframes, static-frames, etc.

Arr: TypeAlias = sparse.SparseArray | sparray | np.ndarray
FeatMat: TypeAlias = Bool[Arr, "obs feat"]
SimsMat: TypeAlias = Num[Arr, "feat feat"]


def _gram(X1, X2):
    grammian = X1.T @ X2
    return grammian.toarray() if issparse(grammian) else grammian


def coocur_prob(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """probability of a co-ocurrence per-observation

    Args:
      X: feature matrix     
      pseudocts:  (Default value = 0.5)

    """

    cts = _gram(X, X)
    tot = X.shape[0]
    return pseudocount(pseudocts)(cts, tot)


# def cond_prob(X:FeatMat, pseudocts:PsdCts=0.):
#     cts = X.T@X + pseudocts
#     tot = X.sum(axis=0)+2
#     return cts/tot


def _not(X):
    if issparse(X):
        nX = X < coo_array(np.ones(X.shape))
    else:
        nX = 1 - X
    return nX


def _marginal_cts(X):
    return X.sum(axis=0), _not(X).sum(axis=0)


def _marginal_prob(X: FeatMat, pseudocts: PsdCts = 0.5):
    # psdct = pseudocount(pseudocts)  #TODO cannot use _sq(num) when num is 1D!!
    # return [psdct(ct, X.shape[0]) for ct in _marginal_cts(X)]
    return [(ct + pseudocts) / (X.shape[0] + 2 * pseudocts) for ct in _marginal_cts(X)]


def _contingency_cts(X):
    nX = _not(X)
    both = _sq(_gram(X, X))
    one = _sq(_gram(X, nX))
    other = _sq(_gram(nX, X))
    neither = _sq(_gram(nX, nX))
    return neither, one, other, both


def _contingency_prob(X: FeatMat, pseudocts: PsdCts = 0.5):
    psdct = pseudocount(pseudocts)
    cts = np.vstack(_contingency_cts(X))
    return [psdct(ct, X.shape[0]) for ct in cts]


def _binary_contingency(X):
    n = X.shape[0]
    both = coocur_prob(X)
    neither = coocur_prob(1 - X)
    one = (X.T @ (1 - X) + 1) / (
        n + 2
    )  # np.logical_and.outer(X, ~X).mean(axis=2).mean(axis=0)
    other = one.T
    return neither, one, other, both


def odds_ratio(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """Ratio of the odds of a true pos/neg to false pos/neg

    For associations, we replace pos/neg and true/false with
    a=yes/no and b=yes/no.
    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(a * d / (b * c)) + np.eye(X.shape[1])


def mutual_information(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """Mutual Information over binary random variables

    For use in e.g. Chow-Liu Trees

    Args:
        X: feature matrix
        pseudocts: (Default value = 0.5) Assumed to apply to contingency table cts

    Returns:

    """
    Pxy = np.array(_contingency_prob(X, pseudocts=pseudocts))

    # Pxy = np.vstack([sq(i) for i in cond_table])
    yes, no = _marginal_prob(X)
    # TODO had to hard-code 0.5 for now :(, pseudocts=pseudocts)

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


def chow_liu(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """Chow-Liu Tree on the features of a (binary) design matrix

    computes mutual information over all pairs of features, and returns
    the maximum spanning tree on them. Assumes a symmetric adjacency is wanted.

    Args:
      X: feature matrix
      pseudocts:  (Default value = 0.5)

    Returns: Adjacency matrix of the Chow Liu MST

    """
    # return sq(sq(minimum_spanning_tree(-MI_binary(X)).todense()))
    return _sq(
        _sq(
            minimum_spanning_tree(
                np.exp(-(mutual_information(X, pseudocts=pseudocts)))
            ).todense()
        )
    )


def yule_y(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """a.k.a. Coefficient of Colligation.

    mobius transform of the Odds Ratio to the range [-1,1]

    Args:
      X: feature matrix
      pseudocts:  (Default value = 0.5)

    Returns: square matrix containing Yule's Y

    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(((ad := np.sqrt(a * d)) - (bc := np.sqrt(b * c))) / (ad + bc))
    # return ((sor:=np.sqrt(OR))-1)/(sor+1)
    # return ((ad:=np.sqrt(both*neither))-(bc:=np.sqrt(one*other)))/(ad+bc)


def yule_q(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """a.k.a. Goodman & Kruskal's gamma for 2x2.

    mobius transform of the Odds Ratio to the range [-1,1]

    Args:
      X: feature matrix
      pseudocts:  (Default value = 0.5)

    Returns: square matrix containing Yule's Q

    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(((ad := a * d) - (bc := b * c)) / (ad + bc))


def ochiai(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    r"""AKA cosine similarity on binary sets

    This code illustrates the idea that we can interpret it as conditional probability:
    The "exposure" of pairwise co-occurrences can't be larger than the sample size,
    so instead we approximate it as a psuedo-variable  having the geometric average of
    two original (conditional) exposure rates

    .. math::
        \sqrt{x_{ii},x_{jj}}

    This interpretation has a nice side-effect of letting us "smooth" the measure with
    laplace/additive pseudocounts on each bernoulli(-ish) "co-occurrence variable".

    Args:
      X: feature matrix
      pseudocts:  (Default value = 0.5)

    Returns: square cosine similarity matrix (incl. ones in the diagonal)

    """
    # I = np.eye(X.shape[-1])
    co_occurs = _sq(_gram(X, X))  # + pseudocts
    exposures = X.sum(axis=0)
    pseudo_exposure = _sq(np.sqrt(_outer(np.multiply, exposures)))  # + 2 * pseudocts
    # return _sq(co_occurs) / pseudo_exposure + np.eye(X.shape[1])
    return _sq(pseudocount(pseudocts)(co_occurs, pseudo_exposure)) + np.eye(X.shape[1])


def binary_cosine_similarity(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """alias of ochiai(X), provided for user convenience

    Args:
      X: feature matrix
      pseudocts:  (Default value = 0.5)

    Returns: cosine similarity on binary feature vectors

    """
    return ochiai(X, pseudocts=pseudocts)


def resource_project(
    X: FeatMat, pseudocts: PsdCts = 0.5, sym_func=np.maximum
) -> SimsMat:
    """bipartite project due to Zhao et al.

    Really just a step of sinkhorn-knopp on the bipartite adjacency
    https://doc.rero.ch/record/8503/files/zhang_bnp.pdf

    For additive smoothing to work, we assume no smoothing is needed
    for the "forward" projection (agents->artifacts), since we assume
    no artifact has 0 agent participation, while some known agents may
    have 0 (observed) artifact participation.

    by default, we symmetrize with "maximum", meaning that association is
    considered as the strongest of the directions it could take. This
    can be overridden with any function of two same-shaped arrays.

    Args:
      X: featiure matrix
      pseudocts:  (Default value = 0.5)
      sym_func:  (Default value = np.maximum)

    Returns: symmetrized "resource projection" similarities

    """
    psdct_func = pseudocount(pseudocts)

    fwd = (X.T / X.sum(axis=1)).T  # right-stochastic bipartite
    # bwd = psdct_func(X, X.sum(axis=0))

    # P = _gram(((X.T) / (X.sum(axis=1))).T, ((X) / (X.sum(axis=0))))
    num = _gram(X, fwd)  # project back
    # den = np.multiply.outer(X.sum(axis=0), np.ones(X.shape[1]))
    den = X.sum(axis=0)  # normalized by node occurrences (bipartite degree)
    P = psdct_func(num, den)
    return sym_func(P, P.T)


def high_salience_skeleton(X: FeatMat, prior=ochiai, pseudocts: PsdCts = "min-connect"):
    """Backboning technique from Grady et al. (2012)
    Calculates shortest paths from every node, and counts the
    number of trees each edge ended up being used in.

    Args:
      X: feature matrix
      prior:  (Default value = ochiai) callable to calculate distances for shortest paths
      pseudocts:  (Default value = "min-connect")

    Returns: (smoothed/beta bernoulli) parameters for shortest path occurrences.

    """
    est_dists = np.abs(-np.log(prior(X, pseudocts=pseudocts)))
    d, pred = shortest_path(est_dists, return_predecessors=True)
    E_obs = np.array(
        [
            _sq(
                _sparse_directed_to_symmetric(
                    reconstruct_path(d, p, directed=False).astype(bool)
                ).toarray()
            )
            for p in pred
        ]
    )
    hss = pseudocount(pseudocts)(E_obs.sum(axis=0), E_obs.shape[0])
    # hss = (E_obs.sum(axis=0) + pseudocts) / (E_obs.shape[0] + 2 * pseudocts)
    return _sq(hss)


def _pursue_tree_basis(dists, nodes, edge_priors=False, beta=0.001):
    N = dists.shape[0]
    all_E = complete_edgelist_on_nodes(N, nodes)
    subset_dists = dists[nodes].T[nodes].T
    if edge_priors:
        # if we are recieving an estimate for the Graph Laplacian
        # ...approximate steiner with local shortest path dists. 
        subset_dists = adjusted_forest_dists(subset_dists, beta=beta)
    tree = sparse.COO.from_scipy_sparse(minimum_spanning_tree(subset_dists))
    tree_E = sq_ij_e(tree.shape[0], tree.coords)
    return all_E[tree_E]


def _spanning_forests_obs_bootstrap(X, prior_dists=None, edge_priors=False, beta=0.001):
    """resample with a kernel bootstrap on MSTs in a manifold"""
    if (prior_dists is None) and (not edge_priors):
        prior_dists = -np.log(ochiai(X, pseudocts=0.5))

    elif edge_priors:
        # TODO: enforce laplacian!
        assert np.allclose(prior_dists.sum(axis=0),0)
        
    # N_obs = X.toarray() if issparse(X) else X
    N_obs = sparse.COO.from_scipy_sparse(X) if issparse(X) else sparse.COO(X)
    N_activations = groupby_col0(N_obs.coords.T)
    E_activations = [
        _pursue_tree_basis(prior_dists, nodes, edge_priors=edge_priors, beta=beta)
        for nodes in N_activations
    ]

    E_coords = (
        np.repeat(np.arange(N_obs.shape[0]), np.array([len(e) for e in E_activations])),
        np.concatenate(E_activations),
    )
    n = X.shape[1]
    m = n * (n - 1) // 2
    return coo_array(
        sparse.COO(E_coords, data=1, shape=(X.shape[0], m)).to_scipy_sparse()
    )

    # E_obs = coo_array(
    #     [
    #         _sq(
    #             minimum_spanning_tree(_get_masked_sq(_sq(prior_dists), i)).todense() > 0
    #         )
    #         for i in N_obs
    #     ]
    # )
    # return E_obs



def forest_pursuit_cts(X: FeatMat, prior_dists=None) -> SimsMat:
    """Point estimate for number of actual edge activations, rather than
    node-node co-occurrences.
    Uses the Empirical Bayes estimate of the Spanning Forest Density

    Args:
      X: feature matrix
      prior_dists:  (Default value = None)

    Returns: counts for approximate steiner tree occurrences.

    """

    # est_dists = -np.log(prior(X, pseudocts=pseudocts))
    e_obs = _spanning_forests_obs_bootstrap(X, prior_dists=prior_dists)
    return _sq(e_obs.sum(axis=0))

def expected_forest_maximization(
    X: FeatMat,
    prior_struct=None,
    beta=0.001,
    eps=1e-5,
    max_iter=100,
    verbose=False,
) -> SimsMat:
    """ Expectation Maximization Scheme to recover structure
    
    """
    if prior_struct is None:
        e_prob = _sq(forest_pursuit_edge(X))
        prior_struct = _norm_diag(edge_weights_to_laplacian(e_prob))

    uv_cts = _sq(_gram(X,X))    
    
    diff, it = 1.,0

    while (diff>eps) and (it<max_iter):
        it+=1
        e_cts = _spanning_forests_obs_bootstrap(X, prior_dists=prior_struct, edge_priors=True, beta=beta).sum(axis=0)
        e_prob_new = e_prob + (e_cts - e_prob*uv_cts)/(uv_cts+1)  # posterior for Beta(a, 1-a)
        diff = np.max(np.abs(e_prob_new - e_prob))
        e_prob = e_prob_new
        prior_struct = _norm_diag(edge_weights_to_laplacian(e_prob))

    if verbose:
        return _sq(e_prob), it
    else:
        return _sq(e_prob)
        

def forest_pursuit_edge(
    X: FeatMat, prior_dists=None, pseudocts: PsdCts = "min-connect"
) -> SimsMat:
    """point estimate for edge-activation probability, conditional on
    both nodes being a priori activated.
    Uses the Spanning Forest Density non-parametric estimator

    Args:
      X: feature matrix
      prior_dists:  (Default value = None) default uses -log(cos-sim)
      pseudocts:  (Default value = "min-connect")

    Returns: probability of edge traversal given a co-occurrence. 
    """
    e_cts = _sq(forest_pursuit_cts(X, prior_dists=prior_dists))
    uv_cts = _sq(_gram(X, X))
    # e_prob = (e_cts + pseudocts) / (uv_cts + 2 * pseudocts)
    e_prob = pseudocount(pseudocts)(e_cts, uv_cts)
    return _sq(e_prob)


def forest_pursuit_interaction(
    X: FeatMat,
    prior_dists=None,
    precalc_prob: SimsMat | None = None,
    pseudocts: PsdCts = "min-connect",
) -> SimsMat:
    """point estimate for probability of observing an edge traversal,
    using the Spanning Forest Density non-parametric estimator. Weights conditional
    edge traversal probability by the base co-occurrence probability. 

    If you have already calculated forest_pursuit_edge, you can pass it as precalc_prob. 

    Args:
      X: feature matrix
      prior_dists: (Default value = None) default uses -log(cos-sim)
      precalc_prob: (Default value = None) to avoid re-computing edge prob
      pseudocts:  (Default value = "min-connect")

    Returns: probability of observing an edge traversal

    """
    # TODO: check X and precalc_cts have correct dim
    if not precalc_prob:
        e_prob = _sq(
            forest_pursuit_edge(X, prior_dists=prior_dists, pseudocts=pseudocts)
        )
    else:
        e_prob = _sq(precalc_prob)
    uv_prob = _sq(coocur_prob(X, pseudocts=pseudocts))
    e_margP = e_prob * uv_prob
    return _sq(e_margP)


def forest_pursuit_normdegs(X, prior_dists=None, pseudocts="min-connect"):
    """DEPRECATED re-scaling of SFD Interraction probability that tends to
    move the highest-probability edges for each node toward 1.0.

    If a graph generating random walks has an underlying "simple, connected"
    structure (unweighted), then edges should tend toward "existing" if they
    are the "best possible" edge available to a given node. This function re-weights
    edges away from "interaction" probabilities by the geometric mean of the strongest
    interraction probabilities for each node, individually.

    I.e. even if an edge is somewhat rare because the node connected to it has many other
    edges (high-degree), our belief in its existence is weighted by its comparative strength of other interactions its two nodes experience.
    """
    e_margP = _sq(
        forest_pursuit_interaction(X, prior_dists=prior_dists, pseudocts=pseudocts)
    )
    gmean_max = _sq(_outer(np.multiply, np.sqrt(_sq(e_margP).max(axis=0))))
    return _sq(e_margP / gmean_max)

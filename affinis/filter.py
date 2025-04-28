import numpy as np
from numpy import ma
import scipy.sparse as sp
from .utils import edge_mask_to_laplacian, _binary_search_greatest, n_nodes_from_edges, _sq
# from functools import partial


def check_connected(L):
    """Uses sparse routine to calculate first two eigenvalues (smallest).
    the second-smallest (Fiedler) is non-zero iff the graph is connected."""
    # p_thres = p * (p > thres)
    # L = laplacian(p_thres)
    sp.coo_array(L)
    connectivity = sp.linalg.eigsh(L, k=2, return_eigenvectors=False, which="SM")
    return not np.allclose(connectivity, [0, 0])


def threshold_edges_filter(edge_weights, thres):
    """threshold edgeweights and return a masked array"""
    return ma.masked_less(edge_weights, thres)


def min_connected_filter(edge_weights):
    """threshold-filter edge-weights until the graph is (just) connected."""

    # ds = sinkhorn(adj)
    # ds_edges = edgelist(ds)
    n = n_nodes_from_edges(edge_weights)
    def connected(thres):
        """bfs cardinality is 100x faster and more stable than eigsh/fiedler"""
        # return check_connected(
        #     edge_mask_to_laplacian(threshold_edges_filter(edge_weights, thres))
        # )
        A = sp.coo_array(_sq(edge_weights>thres))
        bfs = sp.csgraph.breadth_first_tree(A, 0, directed=False)
        return bfs.sum() == n - 1

    pos = edge_weights.argsort()

    min_idx, min_thres = _binary_search_greatest(
        edge_weights[pos],
        connected,
        0,
        len(edge_weights) - n + 1,  # need at least n-1 edges for connected
    )
    # return adj * (ds >= min_thres)
    # print(pos[min_idx], min_thres)
    return threshold_edges_filter(edge_weights, min_thres)

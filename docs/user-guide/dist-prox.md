# Distance & Proximity

<!-- #region -->

These modules mostly consist of helper functions for the `affinis.associations` module, but may be of interest in their own right. 

The so-called "forest" kernel is a parameterized form of an inverse regularized Laplacian: 

$$ Q_{\beta} = \left( I+\beta L \right)^{-1} $$


Entries in this proximity matrix turn out to be the probability that a node ends up sharing a tree with another node, in a randomly sampled spanning forest of the graph. (hence the name).
<!-- #endregion -->

```python
from affinis.proximity import forest
from affinis.utils import edge_mask_to_laplacian


L = edge_mask_to_laplacian(filtered).data
# hinton(L)
sns.heatmap(forest(L))
```

Another useful tool in this module is the `sinkhorn` function, which performs iterated proportional fitting (i.e. the Sinkhorn-Knopp iterations) to project a sqare matrix to it's nearest _doubly stochastic_ counterpart (rows and columns all sum to _approximately_ 1)

This might be a useful way to approximate the forest kernel (which happens to also be doubly stochastic, per Chebotarev et al.) when you aren't willing to trust a threshold to be an accurate graph reconstruction.

```python
from affinis.proximity import sinkhorn 

sns.heatmap(sinkhorn(ochiai(X)))

print(sinkhorn(ochiai(X)).sum(axis=1))
```

From these proximities and kernels, we are able to turn them into distance metrics, using the bilinear form (see the documentation for more details).

```python
from affinis.distance import adjusted_forest_dists, generalized_graph_dists,bilinear_dists

sns.heatmap(bilinear_dists(forest(L)))
```

This has been shortened, given a graph Laplacian, to quickly retrive the linear an logarithmic forms of the bilinear distance metric on a graph using `adjusted_forest_dists` and `generalized_graph_dists`, respectively:

```python
sns.heatmap(generalized_graph_dists(L))
```

## Filtering 

It's commonly necessary to select a threshold value over which an edge "exists" and below which it doesn't. 
Since edge recovery is very often an _unsupervised_ problem, a common way to select a relatively sparse threshold value is by removing edges until just before the graph would become disconnected. 

`affinis` has implemented a fast routine for removing edges until the graph is about to become disconnected (using a binary search and breadth-first connectivity checks). 
The returned array is a `numpy.ma.masked_array`, which is useful for simultaneously storing the sparsity pattern and the un-filtered values.

```python
from affinis.filter import min_connected_filter

filtered = min_connected_filter(unwrap(forest_pursuit_edge(X)))


filtered
```

We can then use our filter on e.g. the Forest Pursuit edge probability estimates to recreate a possible MRF (adjacency matrix) that could have generated the observed occurence structure:

```python
A = unwrap(~filtered.mask)
hinton(A)
```

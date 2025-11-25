---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python (affinis)
  language: python
  name: affinis
---

# Using `affinis` for Relational Analysis

```{code-cell} ipython3
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# rng = np.random.default_rng(42)
sns.set_theme(style='white')
```

```{code-cell} ipython3
B = nx.bipartite.random_graph(15,30, .25, seed=2)
n = list(B.nodes)[15:]
```

```{code-cell} ipython3
X = nx.bipartite.biadjacency_matrix(B, n).toarray()
plt.spy(X)
plt.axis('off')
```

```{code-cell} ipython3
X.sum(axis=1)
```

## Visualizing Relations

```{code-cell} ipython3
cooc = X.T@X

plt.spy(cooc, marker='o', color='k')
plt.axis('off')
```

```{code-cell} ipython3
from affinis.plots import hinton

hinton(cooc)
```

```{code-cell} ipython3
sns.heatmap(cooc)
```

## Measuring Relatedness

+++

### Basic Association

```{code-cell} ipython3
# from affinis.utils import sq_e_ij, _sq
from scipy.spatial.distance import squareform

def unwrap(A):
    return squareform(A, checks=False)
    
```

```{code-cell} ipython3
cooc = X.T@X

unwrap(cooc)
```

```{code-cell} ipython3
from affinis.associations import coocur_prob, ochiai, odds_ratio, yule_y, yule_q

# sns.histplot(unwrap(cooc_probs))
sns.heatmap(coocur_prob(X, pseudocts=0.))
```

```{code-cell} ipython3
sns.heatmap(ochiai(X, pseudocts=0.), vmin=0)
```

### Additive Smoothing

```{code-cell} ipython3
cos_sim = ochiai(X, pseudocts=1.)
sns.heatmap(cos_sim, vmin=0)
```

```{code-cell} ipython3
for i in [0., 0.5, 1.]: 
    sns.histplot(unwrap(ochiai(X, pseudocts=i)), label=i, element='step')
plt.legend()
```

### Backboning Methods


Going a bit further than pure association methods, for bipartite projection problems like these we are often trying to "backbone" our graph (association measures are notorious for turning these into "hairballs"). An example of this is the so-called _High-Salience Skeleton_. 


Alternatively, we might try to discover an underlying probabilistic graphical model that could have generated our observations. For example, the Chow-Liu True, or Forest Pursuit edge probability estimates.

```{code-cell} ipython3
from affinis.associations import high_salience_skeleton, chow_liu, forest_pursuit_edge
```

```{code-cell} ipython3
f,(ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(16,4))
sns.heatmap(high_salience_skeleton(X, prior=coocur_prob), square=True, ax=ax1, )
sns.heatmap(chow_liu(X), square=True, ax=ax2)
sns.heatmap(forest_pursuit_edge(X), square=True, ax=ax3)
ax1.set_title('HSS Backbone')
ax2.set_title('Chow-Liu Tree')
ax3.set_title('Forest Pursuit Edge Prob.');
```

There are many more ways to think about edge recovery, which are covered in a bit more detail in _Measuring Node Activations_ (Sexton 2025)

+++ {"jp-MarkdownHeadingCollapsed": true}

### Overview

![Table 4.1 from "Measuring Network Dependencies from Node Activations" (Sexton 2025)](https://dissertation.rtbs.dev/content/part1/1-03-recovery-road_files/figure-html/-content-codefigs-graphs-tbl-roads-output-1.svg)

+++

## Filtering 

It's commonly necessary to select a threshold value over which an edge "exists" and below which it doesn't. 
Since edge recovery is very often an _unsupervised_ problem, a common way to select a relatively sparse threshold value is by removing edges until just before the graph would become disconnected. 

`affinis` has implemented a fast routine for removing edges until the graph is about to become disconnected (using a binary search and breadth-first connectivity checks). 
The returned array is a `numpy.ma.masked_array`, which is useful for simultaneously storing the sparsity pattern and the un-filtered values.

```{code-cell} ipython3
from affinis.filter import min_connected_filter

filtered = min_connected_filter(unwrap(forest_pursuit_edge(X)))


filtered
```

We can then use our filter on e.g. the Forest Pursuit edge probability estimates to recreate a possible MRF (adjacency matrix) that could have generated the observed occurence structure:

```{code-cell} ipython3
A = unwrap(~filtered.mask)
hinton(A)
```

## Proximity & Distance

These modules mostly consist of helper functions for the `affinis.associations` module, but may be of interest in their own right. 

The so-called "forest" kernel is a parameterized form of an inverse regularized Laplacian: 

$$ Q_{\beta} = \left( I+\beta L \right)^{-1} $$


Entries in this proximity matrix turn out to be the probability that a node ends up sharing a tree with another node, in a randomly sampled spanning forest of the graph. (hence the name).

```{code-cell} ipython3
from affinis.proximity import forest
from affinis.utils import edge_mask_to_laplacian


L = edge_mask_to_laplacian(filtered).data
# hinton(L)
sns.heatmap(forest(L))
```

Another useful tool in this module is the `sinkhorn` function, which performs iterated proportional fitting (i.e. the Sinkhorn-Knopp iterations) to project a sqare matrix to it's nearest _doubly stochastic_ counterpart (rows and columns all sum to _approximately_ 1)

This might be a useful way to approximate the forest kernel (which happens to also be doubly stochastic, per Chebotarev et al.) when you aren't willing to trust a threshold to be an accurate graph reconstruction.

```{code-cell} ipython3
from affinis.proximity import sinkhorn 

sns.heatmap(sinkhorn(ochiai(X)))

print(sinkhorn(ochiai(X)).sum(axis=1))
```

From these proximities and kernels, we are able to turn them into distance metrics, using the bilinear form (see the documentation for more details).

```{code-cell} ipython3
from affinis.distance import adjusted_forest_dists, generalized_graph_dists,bilinear_dists

sns.heatmap(bilinear_dists(forest(L)))
```

This has been shortened, given a graph Laplacian, to quickly retrive the linear an logarithmic forms of the bilinear distance metric on a graph using `adjusted_forest_dists` and `generalized_graph_dists`, respectively:

```{code-cell} ipython3
sns.heatmap(generalized_graph_dists(L))
```

```{code-cell} ipython3

```

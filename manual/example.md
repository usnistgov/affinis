---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.20.0
kernelspec:
  name: affinis
  display_name: Python (affinis)
  language: python
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Example: Reconstructing Colleague Networks with Forest Pursuit

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
sns.set_theme(style='white')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem Setting

Synthesizing a network of colleagues that ask each other to join them on papers:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
n_authors=25
author_idx = pd.CategoricalIndex((f'author_{i:>02}' for i in range(1,n_authors+1)))

# friends with some cliques
friendships = nx.line_graph(nx.random_labeled_tree(len(author_idx)+1, seed=7)) # real events... what "happens" as evidence of a relationship

G = nx.relabel.relabel_nodes(
    nx.convert_node_labels_to_integers(friendships),
    dict(zip(range(n_authors),author_idx.categories.tolist()))
)  # inferred structure


A = nx.adjacency_matrix(G).todense()
L = nx.laplacian_matrix(G).todense()


def draw_G(G, ax=None):
    
    pos=nx.layout.kamada_kawai_layout(G)
    nx.draw(G, pos=pos, 
            node_color='xkcd:puce', edge_color='grey', ax=ax)
    nx.draw_networkx_labels(G, pos=pos, font_color='k',labels={n:n.split('_')[-1] for n in G}, ax=ax)
    plt.title('Author Friendships', color='grey')
    return pos
f = plt.figure(figsize=(5,4)).patch.set_alpha(0.)
pos = draw_G(G)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Alternatively, we can view this network as an adjacency matrix:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from affinis.plots import hinton
hinton(A)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Now we need to simulate the process of authors joining each list of authors. 

- for 50 papers, we select a random individual to intiate it. 
- for each paper, we spread the paper's concept to colleagues.
  - a geometrically distributed number of requests to join will be successful,
  - Each request comes from an existing author, able to ask any of their connected colleagues to join
- Represent the authors on a given paper each week as "active"

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can visualize these author-paper connections as binary "activation" relationships in a matrix, with one author-per-column, one paper-per-row:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
def sim_papers(n_weeks, L, jumps_param=0.1, rng=np.random.default_rng(2)): 
    Arw = ((L/np.diag(L)).pipe(lambda df: np.diag(np.diag(df))-df)*0.5)
    def sim_week(): 
        n_jumps = rng.geometric(jumps_param)
        first = rng.multinomial(1,starting:=np.ones(n_authors)/n_authors) 
        # second = (rng.random()>0.5)*rng.multinomial(1,starting)  # maybe
        infected = first #| second
        for jump in range(n_jumps):
            # print((Arw@infected>1).sum(), infected)
            infected = infected | rng.binomial(1, Arw@(infected/infected.sum()))
        return infected


    yield from (sim_week() for i in range(n_weeks))

# n_obs ~ neg_binom(2, 1/n_nodes)
# n_jumps ~ geom(2/n_nodes)
X = np.vstack(list(sim_papers(
    52, 
    pd.DataFrame(L, columns=author_idx, index=author_idx), 
    0.05,
    #  rng=rng
)))
Xdf = pd.DataFrame(X, columns=author_idx)

# Xstack = np.vstack([X, -X])#.mean(axis=0)

hinton(X)
# plt.axis('off');
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Number of papers each author participated on:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plt.figure(figsize=(4,2))
sns.histplot(Xdf.sum(axis=0), discrete=True)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["hide-input", "remove-stderr"]}

Number of authors on each paper:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, remove-stderr]
---
plt.figure(figsize=(4,2))
sns.histplot(Xdf.sum(axis=1), discrete=True)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Association functions

The core of the `affinis` library lives within the `affinis.associations` module.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from affinis.associations import coocur_prob, ochiai
from affinis.plots import hinton

cooc = coocur_prob(X, pseudocts=0.)

csim = ochiai(X, pseudocts=0.)

hinton(cooc)
hinton(L)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
hinton(csim)
plt.spy(L, marker='.', color='r')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from affinis.associations import (
    coocur_prob,
    odds_ratio,
    mutual_information,
    chow_liu,
    yule_q, yule_y,
    ochiai,
    resource_project,
    high_salience_skeleton
)

from affinis.utils import (
    _norm_diag,
    # _e_to_ij, 
    # _std_incidence_vec, 
    _sq, 
    _outer,
    sparse_adj_to_incidence,
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def prox_to_laplacian(K):
    A = -_sq(_sq(K))
    np.fill_diagonal(A,-A.sum(axis=0))
    return A

psct ='min-connect'
# psct=0

baselines = {
    'cosine': ochiai(X, pseudocts=psct),
    'resourceProj': resource_project(X),
    'odds-ratio': odds_ratio(X, pseudocts=psct),
    'yuleY':yule_y(X, pseudocts=psct),
    'mutualinfo': mutual_information(X, pseudocts=psct),
    'HSS': high_salience_skeleton(X)
}
f,axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6))

for n, (lab, Aest) in enumerate(baselines.items()): 
    ax = axs.flatten()[n]
    # ax.imshow(Aest)
    hinton(Aest, ax=ax)
    ax.set_xlabel(lab)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from affinis.associations import forest_pursuit_edge

hinton(forest_pursuit_edge(X, pseudocts=psct))
# plt.spy(A, marker='.', color='r')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from contingency import Contingent
from contingency.plots import PR_contour

m_fp = Contingent.from_scalar(_sq(A).astype(bool), _sq(forest_pursuit_edge(X, pseudocts=psct)))
m_mi = Contingent.from_scalar(_sq(A).astype(bool), _sq(baselines['mutualinfo']))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plt.plot(m_fp.weights, m_fp.mcc)
plt.plot(m_mi.weights, m_mi.mcc)

m_fp.expected('mcc'), m_mi.expected('mcc')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
PR_contour()
plt.step(m_fp.recall, m_fp.precision, where='post')
plt.step(m_mi.recall, m_mi.precision, where='post')
m_fp.expected('aps'), m_mi.expected('aps')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```

```{code-cell} ipython3

```

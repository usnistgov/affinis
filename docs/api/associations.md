All functions in this module take in data as boolean design matrices (i.e. observations x features), and return a feature association measure (i.e. features x features). 

$$
f: \mathbb{B}^{i\times j} \rightarrow \mathbb{R}^{j\times j}
$$

Note that some of these functions return valid adjacency matrices (e.g. a feature is not associated to itself), while others return covariance or correlations (features are partially or fully correlated to themselves). 
Many of the more basic association measures are given in terms of conditional probabilities via a contingency table, for which we adopt the following notation, where A and B are individual feature columns of X:

$$
\begin{array}{c|cc|c}
      & B=1         & B=0         & \sum_B \\
\hline 
A=1   & p_{11}      & p_{10}      & p_{1\bullet} \\
A=0   & p_{01}      & p_{00}      & p_{0\bullet} \\
\hline 
\sum_A   & p_{\bullet 1} & p_{\bullet 0} & p_\bullet \\
\end{array}
$$ 

Where appropriate, the methods here allow for additive/laplace smoothing through the `psuedocounts` parameter, even in cases where this is not traditionally done (e.g. cosine similarity).
We give interpretations of meaning that allow for this, where we can.

::: affinis.associations
    options: 
      parameter_headings: false
      members_order: [__all__]
      filters:
        - "!^_"
        - "^__"
        - "!forest_pursuit_cts"
        - "!forest_pursuit_edge"
        - "!forest_pursuit_interaction"
        - "!expected_forest_maximization"


## Forest Pursuit Modes

For the sake of completeness, we provide the underlying functions for each of the `forest_pursuit` modes, here. 

::: affinis.associations.forest_pursuit_cts

::: affinis.associations.forest_pursuit_edge

::: affinis.associations.forest_pursuit_interaction

::: affinis.associations.expected_forest_maximization


All functions in this module take in data as design matrices (i.e. observations x features), and return a feature association measure (i.e. features x features). 

Note that some of these functions return valid adjacency matrices (e.g. a feature 
is not associated to itself), while others return covariance or correlations (features 
are partially or fully correlated to themselves). 

Where appropriate, the methods here allow for additive/laplace smoothing, even in cases 
where this is not traditionally done (like cosine similarity). We give interpretations
of meaning that allow for this, where we can.

::: affinis.associations
    options: 
      parameter_headings: false
      members_order: [__all__]

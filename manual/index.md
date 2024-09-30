# Affinis
> Tools for inferring relations from co-occurrence data

`Affinis` is a tool for assisting in unsupervised _structure learning_ on sparse, binary data. 

## Quick-links

```{tableofcontents}
```

## What does it help with?

```{sidebar}
E.g. given a bag-of-words matrix (a type of NLP embedding) figure out how the tokens/concepts (columns) in the corpus are related to each other, using only the set of documents (rows) that record token co-occurrences in them. 
```

In large (sparse) feature matrices, especially ones with binary or integer-valued entries, you commonly need to figure out the underlying structure of your feature space from the observations. 

Techniques for this are widely varied, and different communities have widely different practices and assumptions for what is an appropriate approach. 

## What's inside?

Primarily, the library's "killer features" live in the `associations` module. 
Here you will find functions collected from a wide variety of disciplines that accept a feature matrix $X$ with $n$ features (columns), and return $n\times n$ square matrices with association measures. 
Other things to see: 

  ```{sidebar} Forest Pursuit
  is lazy, trivially parallelizable, and scales nearly linearly with the size of the feature matrix for diffusion-like problems (worst-case quadratic, otherwise).  
  ```

- reference implementations of our new _Forest Pursuit_ algorithm, 
- universal smoothing api: use `pseudocts=` for easy application of Beta-Binomial prior!
- makes use of new `sparse` library to avoid full instantiation of $X$ in memory
- plotting utilities (vectorized implementation of _Hinton_ diagrams)
- linear-algebra-based graph utilities, 
  - edge probability in random spanning trees/forests, 
  - minimum-connectivity graph weight thresholding, 
  - closed-form edge-to-node-pair index mapping for undirected graph edge subsampling
- WIP: gibbs-sampling technique for fully bayesian semiparametric edge probability estimation

and a lot more. 





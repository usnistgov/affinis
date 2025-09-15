# Affinis
> Tools for inferring relations from binary co-occurrence data

`Affinis` is a library of tools for assisting in unsupervised _structure learning_ on sparse, binary data. 

## What does it help with?


In large (sparse) feature matrices, especially ones with binary or integer-valued entries, you commonly need to figure out the underlying structure of your feature space from the observations. 

> _E.g. given a bag-of-words matrix (a type of NLP embedding) figure out how the tokens/concepts (columns) in the corpus are related to each other, using only the set of documents (rows) that record token co-occurrences in them._

Techniques for this are widely varied, and different communities have widely different practices and assumptions for what is an appropriate approach. 
_Affinis_ provides a library of implementations---_with a consistent interface_---for approaching this problem. 



## What's inside?

_Affinis_ should be considered a prototype for the purposes of research and community benchmark assistance. 
(approx. TRL 4-5)

Primarily, this library's core features live in the `associations` module. 
Here you will find functions collected from a wide variety of disciplines that accept a feature matrix $X$ with $n$ features (columns), and return $n\times n$ square matrices with association measures. 

### Other things to see: 

- Reference implementations of our new _Forest Pursuit_ algorithm, 
  > Forest Pursuit is lazily executable, trivially parallelizable, and scales approximately linearly with the size of your feature matrix for diffusion-like problems (worst-case quadratic, otherwise).  
- Universal smoothing api: use `pseudocts=` for easy application of Beta-Binomial prior!
- Makes use of new PyData `sparse` library to avoid full instantiation of $X$ in memory
- Plotting utilities (including a vectorized implementation of so-called _Hinton_ diagrams)
- Linear-algebra-based graph utilities, 
  - Edge probability in random spanning trees/forests, 
  - Minimum-connectivity graph weight thresholding, 
  - Closed-form edge-to-node-pair index mapping for undirected graph edge subsampling

### Work-in-Progress:

- Gibbs-sampling technique for fully bayesian semiparametric edge probability estimation

## Installation 

`affinis` is currently awaiting pre-publication review. 
Reference installations can be achieved for development purposes with pip: 

```shell
pip install git+https://github.com/usnistgov/affinis.git
```

## Other Information

### Contact the PI

[Rachael Sexton](https://www.nist.gov/people/rachael-t-sexton)

- [`rachael.sexton@nist.gov`](mailto:rachael.sexton@nist.gov)
- NIST Engineering Laboratory
- Systems Integration Division
- Information Modeling & Testing Group 


### Related Material

- Link to documentation webpage: [WIP]()
- Original work first describing _Forest Pursuit_: [dissertation link](https://dissertation.rtbs.dev) 
- Citation:
  > AWAITING PUBLICATION APPROVAL




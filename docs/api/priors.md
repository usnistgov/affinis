

This module exposes the `pseudocount` (higher-order) function, which the [associations](affinis.associations) module uses for general-purpose additive smoothing.

Note that it returns _another_ function, which itself will take _numerator_ and _denominator_ arrays for applying additive smoothing. 

::: affinis.priors


## A note on `zero-sum` & `min-connect` priors

Combining `('zero-sum','min-connect')` will resuld in `a=2/n`,`b=1-2/n` for n `feat` dimensions.

To understand why, we might want this,  recall that if observations are trials over an array of graph edges, then the number of edges that are on or off in a graph is "zero sum" (one extra "on" means one less "off).
So, the proportion of time we will be observing an "on" edge might be thought of as a Wiener Process, and thus follows a (generalized) arcsine distribution.
This means we need a "bathtub" prior a+b=1.

For a concave beta (a,b < 1), the anti-mode is the least likely spot, with the two(limiting) modes being at 0,1.
If a complete graph has n(n-2)/2 edges, while a min. connected one has n-1, then we can bias toward non-edges such that the least-likely p is the ratio: 

$$
\frac{1-(n-1)}{\frac{n*(n-1)}{2}}
$$

This comes out to `a=2/n, b=1-2/n`, so 

$$
P(p|a,b) = \frac{\textrm{#successes}+2/n}{\textrm{#trials}+1}
$$

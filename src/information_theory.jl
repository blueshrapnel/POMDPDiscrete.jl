@doc raw"""
    entropy

Shannon entropy of a random variable ``H(X)`` is the average amount of information gained when you determine a random variable's value, or the inherent level of uncertainty in the possible outcomes of the variable.

```math
H(X) = -\sum_{x} p(x) \log(p(x))
```
"""

@doc raw"""
    conditional_entropy

Average reduction in uncertainty of Y when observing X.
```math
H(Y|X) = -\sum_{x, y} p(x,y) \log p(y|x)\leq (H(Y))
```
"""

@doc raw"""
    mutual_information

The mutual information ``I(X;Y)`` is the measure of the mutual dependence between the variables X and Y, it quantifies the amount of information obtained about one variable by observing another variable.

```math
I(X;Y) = H(Y) - H(Y|X) = -\sum_{x} p(x)\sum_{y}p(y|x) \log \frac{p(y|x)}{p(y)}
```
"""

@doc raw"""
    conditional_mutual_information

How much information X gives about Y given the knowledge of the value of Z I(X;Y|Z).

    ```math
    I(X;Y) = H(Y) - H(Y|X) = -\sum_{x} p(x)\sum_{y}p(y|x) \log \frac{p(y|x)}{p(y)}
    ```

"""

@doc raw"""
    kullback_leibler_divergence

Divergence between two distributions.  An important use case is that mutual information between two variables is the divergence between their joint probabilities and the product of their marginals.
``` math
    D_{KL}[p(x)\Vert q(x)] = \sum_x p(x) \log \frac{p(x)}{q(x)}
```
"""


@doc raw"""

Rate distortion is a constrained convex optimisation problem
```math
\min_{p(y|x)}I(X;Y)\text{ subject to } D_p\leq \text{ a threshold}

```

"""

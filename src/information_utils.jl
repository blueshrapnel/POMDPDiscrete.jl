@doc raw"""
Information Theory Utilities

Methods to calculate information measures on probability distributions.
See also `InformationMeasures.jl`: @Tchanders on [github](https://github.com/Tchanders/InformationMeasures.jl).

The units of information measures depend on the base of the logarithms applied: logarithm to base 2 results in units of bits, whereas using the natural logarithm (to base e) results in units of nats.  In this work all information measures are reported in units of bits.  According to the Julia documentation, "If b is a power of 2 or 10, log2 or log10 should be used, as these will typically be faster and more accurate. "  Thus in this code we aim to use `log2` and the corresponding `exp2` functions to calculate logarithms and exponentials.

"""

@doc raw"""
We apply the usual convention of setting ``0 \log 0 = 0`` as by continuity, ``x \log x \to 0`` for ``x \to 0`` from above.  Extending this continuity argument,  a similar convention as adopted such that ``0 \log \frac{0}{r} = 0`` and ``p \log \frac{p}{0}=\infty`` for all ``r\geq 0`` and ``s>0``.
"""

# see package LogExpFunctions.jl
# https://github.com/JuliaStats/LogExpFunctions.jl/blob/master/src/basicfuns.jl
function xlog2x(x::Number)
    result = x * log2(x)
    return iszero(x) ? zero(result) : result
end

function xlog2y(x::Number, y::Number)
    result = x * log2(y)
    return iszero(y) || (iszero(x) && !isnan(y)) ? zero(result) : result
end


@doc raw"""
    compute_entropy(pX)

Shannon entropy of a random variable ``H(X)`` is the average amount of information gained when you determine a random variable's value, or the inherent level of uncertainty in the possible outcomes of the variable.

```math
H(X) = -\sum_{x} p(x) \log(p(x))
```
"""

function compute_entropy(pX)
    # assume that sum(pX) == 1, avoid performance hit on defensive programming
    # another option could be to use a Distribution or other relevant type
    # @assert sum(pX) ≈ 1
    entropy = -sum(xlog2x.(pX))
    # Make sure we don't return -0.0
	return entropy + 0.0
end



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

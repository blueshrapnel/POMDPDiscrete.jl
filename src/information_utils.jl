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
    entropy(pX)

Shannon entropy of a random variable ``H(X)`` is the average amount of information gained when you determine a random variable's value, or the inherent level of uncertainty in the possible outcomes of the variable.  The same method can be used to calculate the entropy of the joint distribution of a pair of random variables.

```math
H(X) = -\sum_{x} p(x) \log(p(x))
H(X, Y) = -\sum_{x, y} p(x,y) \log(p(x,y))
```

pX : defines p(x), shape (dim_X, )
"""

function entropy(pX)
    # assume that sum(pX) == 1, avoid performance hit on defensive programming
    # another option could be to use a Distribution or other relevant type
    # @assert sum(pX) ≈ 1
    return -sum(xlog2x.(pX))
    # if returning -0.0 is a problem then return entropy + 0.0
end



@doc raw"""
    conditional_entropy (pY_X, pX)

Average reduction in uncertainty of Y when observing X.  This is the expected value of the entropy of the conditional distributions averaged over the conditioning random variable.
```math
\begin{aligned}
H(Y|X) &= -\sum_{x, y} p(x,y) H(Y|X=x) \\
&= -sum_x p(x)\sum_y p(y|x) \log p(y|x) \\
&= \mathbb{E} \log p(Y|X)
\end{aligned}
```
Also, ``H(X,Y) = H(X) + H(Y|X)`` and the corollary, ``H(X,Y|Z) = H(X|Z + H(Y|,X,Z)) .  See Thomas and Cover section 2.2.

pY_X : defines conditional probablity p(Y|x), shape (dim_Y, dim_X)
pX : defines p(x), shape (dim_X, )

"""
function conditional_entropy(pY_X, pX)
    #=
      conditional variable indexed by columns
      assuming p(X,Y) joint distribution with:
      X indexed by column, pX = sum(pXY, dims=1) pX is a row vector, sum columns
      Y indexed by row, pY = sum(pXY, dims=2) pY is a column vector, sum rows
    =#
    pX = convert.(AbstractFloat, pX) |> vec
    HY_X = -sum(xlog2x.(pY_X), dims=1) * pX
   return HY_X[]
end

@doc raw"""
    KL_divergence()

The Kullback-Leibler (KL) Divergence between two distributions, ``p(X)`` and ``q(X)``.  This is also known as the relative entropy.  An important use case is the mutual information between two variables as it is the KL divergence between their joint probabilities and the product of their marginals.
``` math
    \begin{aligned}
    D_{KL}[p(x)\Vert q(x)] &= \mathbb{E}_p[\log \frac{p(x)}{q(x)}]
    &= \sum_x p(x) \log \frac{p(x)}{q(x)}
    \end{aligned}
```

    pX : defines p(x), shape (dim_X, )
    qX : defines q(x), shape (dim_X, )
"""
function KL_divergence(pX, qX)
    # should test that size(pX) == size(qX)
    # potential for divide by zero errors
    KL = sum(xlog2y.(pX, pX./qX))
    return KL
end


@doc raw"""
    mutual_information(pXY)

Compute the mutual information between two random variables, X and Y given either the joint distribution ``p(x,y)`` or the conditional distribution ``p(y|x)`` and the marginal ``p(x)``.

The mutual information ``I(X;Y)`` is the measure of the mutual dependence between the variables X and Y, it quantifies the amount of information obtained about one variable by observing another variable.

```math
I(X;Y) = H(Y) - H(Y|X) = -\sum_{x} p(x)\sum_{y}p(y|x) \log \frac{p(y|x)}{p(y)}
```
pXY : defines the joint distribution p(x,y), shape (dim_Y, dim_X)
"""
# calculate mutual information given the joint distribution
function mutual_information(pXY)
    pX = sum(pXY, dims=1)  # row vector
    pY = sum(pXY, dims=2)  # column vector
    return KL_divergence(pXY, pY * pX)
end

"""
calculate mutual information given a conditional and a marginal distribution
pY_X : defines conditional probablity p(Y|x), shape (dim_Y, dim_X)
pX : defines p(x), shape (dim_X, )
"""
function mutual_information(pY_X, pX)
    pY = pY_X * pX
    pX = convert.(AbstractFloat, pX)
    return sum(xlog2y.(pY_X, pY_X ./pY) * pX)
end

@doc raw"""
    conditional_mutual_information

How much information X gives about Y given the knowledge of the value of Z I(X;Y|Z).


"""


@doc raw"""

Rate distortion is a constrained convex optimisation problem
```math
\min_{p(y|x)}I(X;Y)\text{ subject to } D_p\leq \text{ a threshold}

```

"""

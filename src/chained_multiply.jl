# Chained multiplication (`A * B * C`, `A * B * C * D`) of static matrices.
#
# The order in which a chain of matrix products is evaluated strongly affects
# the number of scalar multiplications that have to be performed.  Generic
# `LinearAlgebra` picks the cheaper of the possible parenthesizations at run
# time from `size(A)` (see https://github.com/JuliaLang/julia/pull/37898).  For
# static arrays the dimensions are part of the type, so the very same cost
# model can be evaluated while the method is being generated and the cheapest
# parenthesization emitted directly.  The resulting code contains no run-time
# branch and the expression that is *not* chosen is never even compiled.
#
# This mirrors `_tri_matmul`/`_quad_matmul` in
# `stdlib/LinearAlgebra/src/matmul.jl`.

# ---------------------------------------------------------------------------
# Cost model (ported from JuliaLang/julia#37898)
# ---------------------------------------------------------------------------

# Number of scalar multiplications needed to evaluate a product tree.  A leaf
# array costs nothing by itself; the cost of `a * b` is the cost of both
# factors plus the cost of the outermost matrix product.
@inline _mul_cost(::Size) = 0
@inline _mul_cost(t::Tuple) = _mul_cost(t[1], t[2])
@inline _mul_cost(a, b) = _mul_cost(a) + _mul_cost(b) +
    prod(_mul_sizes(a)) * last(_mul_sizes(b))

# Static shape of the result of a product tree.  For a leaf this is its `Size`;
# for a subtree `(a, b)` it is `(rows(a), cols(b))`, exactly as in Base.
@inline _mul_sizes(s::Size) = Tuple(s)
@inline _mul_sizes(t::Tuple) = (first(_mul_sizes(t[1])), last(_mul_sizes(t[2])))

# ---------------------------------------------------------------------------
# Compile-time associativity choice
# ---------------------------------------------------------------------------

# Wrap a generated body so that the returned method is always inlined.  Without
# this a generated function is normally emitted as an out-of-line call, which
# adds overhead to these (often very small) multiplications.
_inline_body(expr) = quote
    $(Expr(:meta, :inline))
    $expr
end

# Three factors.  Base compares `(A*B)*C` with `A*(B*C)`; ties go to the left
# (same as Base).
@generated function _mul_chain(a::StaticMatMulLike, b::StaticMatMulLike,
                               c::StaticMatMulLike)
    Sa, Sb, Sc = Size(a), Size(b), Size(c)
    if _mul_cost(Sa, (Sb, Sc)) < _mul_cost((Sa, Sb), Sc)
        return _inline_body(:(a * (b * c)))
    else
        return _inline_body(:((a * b) * c))
    end
end

# Four factors.  All five binary trees are considered.  The scalar
# multiplication counts come from the ported cost model, but the tie-break is
# static-specific: `LinearAlgebra` checks `(AB)(CD)` first, which keeps two
# independent intermediates alive.  For stack-allocated static matrices a
# left-deep tree reuses one accumulator and benchmarks measurably faster
# (e.g. the all-equal `(4x4)^4` case), so among equally cheap trees we prefer
# the left association.  This also matches the 3-factor tie-break above.
@generated function _mul_chain(a::StaticMatMulLike, b::StaticMatMulLike,
                               c::StaticMatMulLike, d::StaticMatMulLike)
    Sa, Sb, Sc, Sd = Size(a), Size(b), Size(c), Size(d)
    c1 = _mul_cost((Sa, Sb), (Sc, Sd))
    c2 = _mul_cost(((Sa, Sb), Sc), Sd)
    c3 = _mul_cost(Sa, (Sb, (Sc, Sd)))
    c4 = _mul_cost((Sa, (Sb, Sc)), Sd)
    c5 = _mul_cost(Sa, ((Sb, Sc), Sd))
    cmin = min(c1, c2, c3, c4, c5)
    if c2 == cmin
        return _inline_body(:(((a * b) * c) * d))
    elseif c1 == cmin
        return _inline_body(:((a * b) * (c * d)))
    elseif c3 == cmin
        return _inline_body(:(a * (b * (c * d))))
    elseif c4 == cmin
        return _inline_body(:((a * (b * c)) * d))
    else
        return _inline_body(:(a * ((b * c) * d)))
    end
end

@inline *(a::StaticMatMulLike, b::StaticMatMulLike, c::StaticMatMulLike) =
    _mul_chain(a, b, c)

@inline *(a::StaticMatMulLike, b::StaticMatMulLike, c::StaticMatMulLike,
          d::StaticMatMulLike) = _mul_chain(a, b, c, d)

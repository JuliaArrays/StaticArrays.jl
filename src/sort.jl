module Sort

import Base: sort, sortperm

using ..StaticArrays
using Base: @_inline_meta
using Base.Order: Forward, Ordering, Perm, Reverse, ord
using Base.Sort: Algorithm, lt

export BitonicSort

struct BitonicSortAlg <: Algorithm end

# For consistency with Julia Base, track their *Sort docstring text in base/sort.jl.
"""
    StaticArrays.BitonicSort

Indicate that a sorting function should use a bitonic sorting network, which is *not*
stable. By default, `StaticVector`s with at most 20 elements are sorted with `BitonicSort`.

Characteristics:
  * *not stable*: does not preserve the ordering of elements which compare equal (e.g. "a"
    and "A" in a sort of letters which ignores case).
  * *in-place* in memory.
  * *good performance* for small collections.
  * compilation time increases dramatically with the number of elements.
"""
const BitonicSort = BitonicSortAlg()


# BitonicSort has non-optimal asymptotic behaviour, so we define a cutoff
# length. This also prevents compilation time to skyrocket for larger vectors.
defalg(a::StaticVector) =
    isimmutable(a) && length(a) <= 20 ? BitonicSort : QuickSort

@inline function sort(a::StaticVector;
              alg::Algorithm = defalg(a),
              lt = isless,
              by = identity,
              rev::Union{Bool,Nothing} = nothing,
              order::Ordering = Forward)
    length(a) <= 1 && return a
    return _sort(a, alg, lt, by, rev, order)
end

@inline function sortperm(a::StaticVector;
               alg::Algorithm = defalg(a),
               lt = isless,
               by = identity,
               rev::Union{Bool,Nothing} = nothing,
               order::Ordering = Forward)
    p = Tuple(axes(a, 1))
    length(a) <= 1 && return SVector{length(a),Int}(p)

    ordr = Perm(ord(lt, by, rev, order), a)
    return SVector{length(a),Int}(_sort(p, alg, isless, identity, nothing, ordr))
end

@inline _sort(a::StaticVector, alg, lt, by, rev, order) =
    similar_type(a)(sort!(Base.copymutable(a); alg=alg, lt=lt, by=by, rev=rev, order=order))

@inline _sort(a::StaticVector, alg::BitonicSortAlg, lt, by, rev, order) =
    similar_type(a)(_sort(Tuple(a), alg, lt, by, rev, order))

@inline _sort(a::NTuple, alg, lt, by, rev, order) =
    sort!(Base.copymutable(a); alg=alg, lt=lt, by=by, rev=rev, order=order)

@inline _sort(a::NTuple, ::BitonicSortAlg, lt, by, rev, order) =
    _bitonic_sort(a, ord(lt, by, rev, order))

# For better performance sorting floats under the isless relation, apply an order-preserving
# bijection to sort them as integers.
@inline function _sort(
    a::NTuple{N, <:Base.IEEEFloat},
    ::BitonicSortAlg,
    lt::typeof(isless),
    by::Union{typeof.((identity, +, -))...},
    rev::Union{Bool, Nothing},
    order,
) where N
    # Exclude N == 2 to avoid a performance regression on AArch64.
    if N > 2 && (order === Forward || order === Reverse)
        _rev = xor(by === -, rev === true, order === Reverse)
        return _intfp.(_bitonic_sort(_fpint.(a), ord(lt, identity, _rev, Forward)))
    end
    return _bitonic_sort(a, ord(lt, by, rev, order))
end

_inttype(::Type{Float64}) = Int64
_inttype(::Type{Float32}) = Int32
_inttype(::Type{Float16}) = Int16

_floattype(::Type{Int64}) = Float64
_floattype(::Type{Int32}) = Float32
_floattype(::Type{Int16}) = Float16

# Modified from the _fpint function added to base/float.jl in Julia 1.7. This is a strictly
# increasing function with respect to the isless relation. `isless` is trichotomous with the
# isequal relation and treats every NaN as identical. This function on the other hand
# distinguishes between NaNs with different payloads and signs, but this difference is
# inconsequential for unstable sorting. The `offset` is necessary because NaNs (in
# particular, those with the sign bit set) must be mapped to the greatest Ints, which is
# Julia-specific.
@inline function _fpint(x::F) where F
    I = _inttype(F)
    offset = Base.significand_mask(F) % I
    n = reinterpret(I, x)
    return ifelse(n < zero(I), n ⊻ typemax(I), n) - offset
end

# Inverse of _fpint.
@inline function _intfp(n::I) where I
    F = _floattype(I)
    offset = Base.significand_mask(F) % I
    n += offset
    n = ifelse(n < zero(I), n ⊻ typemax(I), n)
    return reinterpret(F, n)
end

# Implementation loosely following
# https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
@generated function _bitonic_sort(a::NTuple{N}, order) where N
    function swap_expr(i, j, rev)
        ai = Symbol('a', i)
        aj = Symbol('a', j)
        return :( ($ai, $aj) = @inbounds lt(order, $ai, $aj) ⊻ $rev ? ($ai, $aj) : ($aj, $ai) )
    end

    function merge_exprs(idx, rev)
        exprs = Expr[]
        length(idx) == 1 && return exprs

        ci = 2^(ceil(Int, log2(length(idx))) - 1)
        # TODO: generate simd code for these swaps
        for i in first(idx):last(idx)-ci
            push!(exprs, swap_expr(i, i+ci, rev))
        end
        append!(exprs, merge_exprs(idx[1:ci], rev))
        append!(exprs, merge_exprs(idx[ci+1:end], rev))
        return exprs
    end

    function sort_exprs(idx, rev=false)
        exprs = Expr[]
        length(idx) == 1 && return exprs

        append!(exprs, sort_exprs(idx[1:end÷2], !rev))
        append!(exprs, sort_exprs(idx[end÷2+1:end], rev))
        append!(exprs, merge_exprs(idx, rev))
        return exprs
    end

    idx = 1:N
    symlist = (Symbol('a', i) for i in idx)
    return quote
        @_inline_meta
        ($(symlist...),) = a
        ($(sort_exprs(idx)...);)
        return ($(symlist...),)
    end
end

end # module Sort

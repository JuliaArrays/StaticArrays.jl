import Base.Order: Forward, Ordering, Perm, ord
import Base.Sort: Algorithm, lt, sort, sortperm


struct BitonicSortAlg <: Algorithm end

const BitonicSort = BitonicSortAlg()


# BitonicSort has non-optimal asymptotic behaviour, so we define a cutoff
# length. This also prevents compilation time to skyrocket for larger vectors.
const _bitonic_sort_limit = 20
defalg(a::StaticVector) =
    isimmutable(a) && length(a) <= _bitonic_sort_limit ? BitonicSort : QuickSort

@inline function sort(a::StaticVector;
              alg::Algorithm = defalg(a),
              lt = isless,
              by = identity,
              rev::Union{Bool,Nothing} = nothing,
              order::Ordering = Forward)
    length(a) <= 1 && return a
    ordr = ord(lt, by, rev, order)
    return _sort(a, alg, ordr)
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
    return SVector{length(a),Int}(_sort(p, alg, ordr))
end


@inline _sort(a::StaticVector, alg, order) =
    similar_type(a)(sort!(copyto!(similar(a), a); alg=alg, order=order))

@inline _sort(a::StaticVector, alg::BitonicSortAlg, order) =
    similar_type(a)(_sort(Tuple(a), alg, order))

_sort(a::NTuple, alg, order) = sort!(collect(a); alg=alg, order=order)

# Implementation loosely following
# https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
@generated function _sort(a::NTuple{N}, ::BitonicSortAlg, order) where N
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

module StaticArraysStatisticsExt

import Statistics: mean, median

using Base.Order: Forward, ord
using Statistics: median!, middle

using StaticArrays
using StaticArrays: BitonicSort, _InitialValue, _reduce, _mapreduce, _bitonic_sort_limit, _sort

_mean_denom(a, ::Colon) = length(a)
_mean_denom(a, dims::Int) = size(a, dims)
_mean_denom(a, ::Val{D}) where {D} = size(a, D)

@inline mean(a::StaticArray; dims=:) = _reduce(+, a, dims) / _mean_denom(a, dims)
@inline mean(f::Function, a::StaticArray; dims=:) = _mapreduce(f, +, dims, _InitialValue(), Size(a), a) / _mean_denom(a, dims)

@inline function median(a::StaticArray; dims = :)
    if dims == Colon()
        median(vec(a))
    else
        # FIXME: Implement `mapslices` correctly on `StaticArray` to remove
        # this fallback.
        median(Array(a); dims)
    end
end

@inline function median(a::StaticVector)
    (isimmutable(a) && length(a) <= _bitonic_sort_limit) ||
        return median!(Base.copymutable(a))

    # following Statistics.median
    isempty(a) &&
        throw(ArgumentError("median of empty vector is undefined, $(repr(a))"))
    eltype(a) >: Missing && any(ismissing, a) &&
        return missing
    nanix = findfirst(x -> x isa Number && isnan(x), a)
    isnothing(nanix) ||
        return a[nanix]

    order = ord(isless, identity, nothing, Forward)
    sa = _sort(Tuple(a), BitonicSort, order)

    n = length(a)
    # sa is 1-indexed
    return isodd(n) ?
        middle(sa[n ÷ 2 + 1]) :
        middle(sa[n ÷ 2], sa[n ÷ 2 + 1])
end

end # module

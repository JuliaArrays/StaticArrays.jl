module StaticArraysStatisticsExt

import Statistics: mean

using StaticArrays
using StaticArrays: _InitialValue, _reduce, _mapreduce

_mean_denom(a, ::Colon) = length(a)
_mean_denom(a, dims::Int) = size(a, dims)
_mean_denom(a, ::Val{D}) where {D} = size(a, D)

@inline mean(a::StaticArray; dims=:) = _reduce(+, a, dims) / _mean_denom(a, dims)
@inline mean(f::Function, a::StaticArray; dims=:) = _mapreduce(f, +, dims, _InitialValue(), Size(a), a) / _mean_denom(a, dims)

end # module

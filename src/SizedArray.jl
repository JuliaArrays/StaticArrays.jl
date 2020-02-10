
"""
    SizedArray{Tuple{dims...}}(array)

Wraps an `Array` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction to determine if the number of elements (`length`) match, but the
array may be reshaped.

The aliases `SizedVector{N}` and `SizedMatrix{N,M}` are provided as more
convenient names for one and two dimensional `SizedArray`s. For example, to
wrap a 2x3 array `a` in a `SizedArray`, use `SizedMatrix{2,3}(a)`.
"""
struct SizedArray{S <: Tuple, T, N, M} <: StaticArray{S, T, N}
    data::Array{T, M}

    function SizedArray{S, T, N, M}(a::Array) where {S, T, N, M}
        if length(a) != tuple_prod(S)
            throw(DimensionMismatch("Dimensions $(size(a)) don't match static size $S"))
        end
        if size(a) != size_to_tuple(S)
            Base.depwarn("Construction of `SizedArray` with an `Array` of a different
                size is deprecated. If you need this functionality report it at
                https://github.com/JuliaArrays/StaticArrays.jl/pull/666 .
                Calling `sa = reshape(a::Array, s::Size)` will actually reshape
                array `a` in the future and converting `sa` back to `Array` will
                return an `Array` of shape `s`.", :SizedArray)
        end
        new{S,T,N,M}(a)
    end

    function SizedArray{S, T, N, M}(::UndefInitializer) where {S, T, N, M}
        new{S, T, N, M}(Array{T, M}(undef, size_to_tuple(S)...))
    end
end

@inline SizedArray{S,T,N}(a::Array{T,M}) where {S,T,N,M} = SizedArray{S,T,N,M}(a)
@inline SizedArray{S,T}(a::Array{T,M}) where {S,T,M} = SizedArray{S,T,tuple_length(S),M}(a)
@inline SizedArray{S}(a::Array{T,M}) where {S,T,M} = SizedArray{S,T,tuple_length(S),M}(a)

@inline SizedArray{S,T,N}(::UndefInitializer) where {S,T,N} = SizedArray{S,T,N,N}(undef)
@inline SizedArray{S,T}(::UndefInitializer) where {S,T} = SizedArray{S,T,tuple_length(S),tuple_length(S)}(undef)

@generated function SizedArray{S,T,N,M}(x::NTuple{L,Any}) where {S,T,N,M,L}
    if L != tuple_prod(S)
        error("Dimension mismatch")
    end
    exprs = [:(a[$i] = x[$i]) for i = 1:L]
    return quote
        $(Expr(:meta, :inline))
        a = SizedArray{S,T,N,M}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline SizedArray{S,T,N}(x::Tuple) where {S,T,N} = SizedArray{S,T,N,N}(x)
@inline SizedArray{S,T}(x::Tuple) where {S,T} = SizedArray{S,T,tuple_length(S),tuple_length(S)}(x)
@inline SizedArray{S}(x::NTuple{L,T}) where {S,T,L} = SizedArray{S,T,tuple_length(S),tuple_length(S)}(x)

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::SizedArray) where {SA<:SizedArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:SizedArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
@inline Array(sa::SizedArray) = Array(sa.data)
@inline Array{T}(sa::SizedArray{S,T}) where {T,S} = Array{T}(sa.data)
@inline Array{T,N}(sa::SizedArray{S,T,N}) where {T,S,N} = Array{T,N}(sa.data)

@inline convert(::Type{Array}, sa::SizedArray) = sa.data
@inline convert(::Type{Array{T}}, sa::SizedArray{S,T}) where {T,S} = sa.data
@inline convert(::Type{Array{T,N}}, sa::SizedArray{S,T,N}) where {T,S,N} = sa.data

@propagate_inbounds getindex(a::SizedArray, i::Int) = getindex(a.data, i)
@propagate_inbounds setindex!(a::SizedArray, v, i::Int) = setindex!(a.data, v, i)

SizedVector{S,T,M} = SizedArray{Tuple{S},T,1,M}
@inline SizedVector{S}(a::Array{T,M}) where {S,T,M} = SizedArray{Tuple{S},T,1,M}(a)
@inline SizedVector{S}(x::NTuple{L,T}) where {S,T,L} = SizedArray{Tuple{S},T,1,1}(x)

SizedMatrix{S1,S2,T,M} = SizedArray{Tuple{S1,S2},T,2,M}
@inline SizedMatrix{S1,S2}(a::Array{T,M}) where {S1,S2,T,M} = SizedArray{Tuple{S1,S2},T,2,M}(a)
@inline SizedMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L} = SizedArray{Tuple{S1,S2},T,2,2}(x)

Base.dataids(sa::SizedArray) = Base.dataids(sa.data)

function (::Size{S})(a::Array) where {S}
    Base.depwarn("`Size{S}(a::Array)` is deprecated, use `SizedVector{N}(a)`, `SizedMatrix{N,M}(a)` or `SizedArray{Tuple{S}}(a)` instead", :Size)
    SizedArray{Tuple{S...}}(a)
end


function promote_rule(::Type{<:SizedArray{S,T,N,M}}, ::Type{<:SizedArray{S,U,N,M}}) where {S,T,U,N,M}
    SizedArray{S,promote_type(T,U),N,M}
end

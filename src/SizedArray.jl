
"""
    SizedArray{Tuple{dims...}}(array)

Wraps an `Array` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction to determine if the number of elements (`length`) match, but the
array may be reshaped.

(Also, `Size(dims...)(array)` acheives the same thing)
"""
struct SizedArray{S <: Tuple, T, N, M} <: StaticArray{S, T, N}
    data::Array{T, M}

    function SizedArray{S, T, N, M}(a::Array) where {S, T, N, M}
        if length(a) != tuple_prod(S)
            error("Dimensions $(size(a)) don't match static size $S")
        end
        new{S,T,N,M}(a)
    end

    function SizedArray{S, T, N, M}() where {S, T, N, M}
        new{S, T, N, M}(Array{T, M}(uninitialized, S.parameters...))
    end
end

@inline (::Type{SizedArray{S,T,N}})(a::Array{T,M}) where {S,T,N,M} = SizedArray{S,T,N,M}(a)
@inline (::Type{SizedArray{S,T}})(a::Array{T,M}) where {S,T,M} = SizedArray{S,T,tuple_length(S),M}(a)
@inline (::Type{SizedArray{S}})(a::Array{T,M}) where {S,T,M} = SizedArray{S,T,tuple_length(S),M}(a)

@inline (::Type{SizedArray{S,T,N}})() where {S,T,N} = SizedArray{S,T,N,N}()
@inline (::Type{SizedArray{S,T}})() where {S,T} = SizedArray{S,T,tuple_length(S),tuple_length(S)}()

@generated function (::Type{SizedArray{S,T,N,M}})(x::NTuple{L,Any}) where {S,T,N,M,L}
    if L != tuple_prod(S)
        error("Dimension mismatch")
    end
    exprs = [:(a[$i] = x[$i]) for i = 1:L]
    return quote
        $(Expr(:meta, :inline))
        a = SizedArray{S,T,N,M}()
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline (::Type{SizedArray{S,T,N}})(x::Tuple) where {S,T,N} = SizedArray{S,T,N,N}(x)
@inline (::Type{SizedArray{S,T}})(x::Tuple) where {S,T} = SizedArray{S,T,tuple_length(S),tuple_length(S)}(x)
@inline (::Type{SizedArray{S}})(x::NTuple{L,T}) where {S,T,L} = SizedArray{S,T,tuple_length(S),tuple_length(S)}(x)

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::SizedArray) where {SA<:SizedArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:SizedArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
@inline (::Type{Array})(sa::SizedArray) = sa.data
@inline (::Type{Array{T}})(sa::SizedArray{S,T}) where {T,S} = sa.data
@inline (::Type{Array{T,N}})(sa::SizedArray{S,T,N}) where {T,S,N} = sa.data

@inline convert(::Type{Array}, sa::SizedArray) = sa.data
@inline convert(::Type{Array{T}}, sa::SizedArray{S,T}) where {T,S} = sa.data
@inline convert(::Type{Array{T,N}}, sa::SizedArray{S,T,N}) where {T,S,N} = sa.data

@propagate_inbounds getindex(a::SizedArray, i::Int) = getindex(a.data, i)
@propagate_inbounds setindex!(a::SizedArray, v, i::Int) = setindex!(a.data, v, i)

SizedVector{S,T,M} = SizedArray{Tuple{S},T,1,M}
@inline (::Type{SizedVector{S}})(a::Array{T,M}) where {S,T,M} = SizedArray{Tuple{S},T,1,M}(a)
@inline (::Type{SizedVector{S}})(x::NTuple{L,T}) where {S,T,L} = SizedArray{Tuple{S},T,1,1}(x)

SizedMatrix{S1,S2,T,M} = SizedArray{Tuple{S1,S2},T,2,M}
@inline (::Type{SizedMatrix{S1,S2}})(a::Array{T,M}) where {S1,S2,T,M} = SizedArray{Tuple{S1,S2},T,2,M}(a)
@inline (::Type{SizedMatrix{S1,S2}})(x::NTuple{L,T}) where {S1,S2,T,L} = SizedArray{Tuple{S1,S2},T,2,2}(x)


"""
    Size(dims)(array)

Creates a `SizedArray` wrapping `array` with the specified statically-known
`dims`, so to take advantage of the (faster) methods defined by the static array
package.
"""
(::Size{S})(a::Array) where {S} = SizedArray{Tuple{S...}}(a)


function promote_rule(::Type{<:SizedArray{S,T,N,M}}, ::Type{<:SizedArray{S,U,N,M}}) where {S,T,U,N,M}
    SizedArray{S,promote_type(T,U),N,M}
end

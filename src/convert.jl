(::Type{SA})(x::Tuple{Tuple{Tuple{<:Tuple}}}) where {SA <: StaticArray} =
    throw(DimensionMismatch("No precise constructor for $SA found. Length of input was $(length(x[1][1][1]))."))

@inline (::Type{SA})(x...) where {SA <: StaticArray} = SA(x)
@inline (::Type{SA})(a::StaticArray) where {SA<:StaticArray} = SA(Tuple(a))
@inline (::Type{SA})(a::StaticArray) where {SA<:SizedArray} = SA(a.data)
@propagate_inbounds (::Type{SA})(a::AbstractArray) where {SA <: StaticArray} = convert(SA, a)

# this covers most conversions and "statically-sized reshapes"
@inline convert(::Type{SA}, sa::StaticArray) where {SA<:StaticArray} = SA(Tuple(sa))
@inline convert(::Type{SA}, sa::StaticArray) where {SA<:Scalar} = SA((sa[],)) # disambiguation
@inline convert(::Type{SA}, sa::SA) where {SA<:StaticArray} = sa
@inline convert(::Type{SA}, x::Tuple) where {SA<:StaticArray} = SA(x) # convert -> constructor. Hopefully no loops...

# support conversion to AbstractArray
AbstractArray{T}(sa::StaticArray{S,T}) where {S,T} = sa
AbstractArray{T,N}(sa::StaticArray{S,T,N}) where {S,T,N} = sa
AbstractArray{T}(sa::StaticArray{S,U}) where {S,T,U} = similar_type(typeof(sa),T,Size(sa))(sa)
AbstractArray{T,N}(sa::StaticArray{S,U,N}) where {S,T,U,N} = similar_type(typeof(sa),T,Size(sa))(sa)

# Constructing a Tuple from a StaticArray
@inline Tuple(a::StaticArray) = unroll_tuple(a, Length(a))

@noinline function dimension_mismatch_fail(SA::Type, a::AbstractArray)
    throw(DimensionMismatch("expected input array of length $(length(SA)), got length $(length(a))"))
end

@propagate_inbounds function convert(::Type{SA}, a::AbstractArray) where {SA <: StaticArray}
    @boundscheck if length(a) != length(SA)
        dimension_mismatch_fail(SA, a)
    end

    return _convert(SA, a, Length(SA))
end

@inline _convert(SA, a, l::Length) = SA(unroll_tuple(a, l))
@inline _convert(SA::Type{<:StaticArray{<:Tuple,T}}, a, ::Length{0}) where T = similar_type(SA, T)(())
@inline _convert(SA, a, ::Length{0}) = similar_type(SA, eltype(a))(())

length_val(a::T) where {T <: StaticArrayLike} = length_val(Size(T))
length_val(a::Type{T}) where {T<:StaticArrayLike} = length_val(Size(T))

@generated function unroll_tuple(a::AbstractArray, ::Length{L}) where {L}
    exprs = [:(a[$j]) for j = 1:L]
    quote
        @_inline_meta
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

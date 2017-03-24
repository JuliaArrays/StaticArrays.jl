(::Type{SA})(x::Tuple{Tuple{Tuple{<:Tuple}}}) where {SA <: StaticArray} = error("No precise constructor for $SA found. Length of input was $(length(x[1][1][1])).")

@inline (::Type{SA})(x...) where {SA <: StaticArray} = SA(x)
@inline (::Type{SA})(a::AbstractArray) where {SA <: StaticArray} = convert(SA, a) # Is this a good idea?

# this covers most conversions and "statically-sized reshapes"
@inline convert(::Type{SA}, sa::StaticArray) where {SA<:StaticArray} = SA(Tuple(sa))
@inline convert(::Type{SA}, sa::SA) where {SA<:StaticArray} = sa
@inline convert(::Type{SA}, x::Tuple) where {SA<:StaticArray} = SA(x) # convert -> constructor. Hopefully no loops...

# A general way of going back to a tuple
@inline function convert(::Type{Tuple}, a::StaticArray)
    unroll_tuple(a, Length(a))
end

@inline function convert(::Type{SA}, a::AbstractArray) where {SA <: StaticArray}
    if length(a) != length(SA)
        error("Dimension mismatch. Expected input array of length $(length(SA)), got length $(length(a))")
    end

    return SA(unroll_tuple(a, Length(SA)))
end

length_val(a::T) where {T <: StaticArray} = length_val(Size(T))
length_val(a::Type{T}) where {T<:StaticArray} = length_val(Size(T))

@generated function unroll_tuple(a::AbstractArray, ::Length{L}) where {L}
    exprs = [:(a[$j]) for j = 1:L]
    quote
        @_inline_meta
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

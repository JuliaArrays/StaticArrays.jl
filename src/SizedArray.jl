
"""
    SizedArray{(dims...)}(array)

Wraps an `Array` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction to determine if the number of elements (`length`) match, but the
array may be reshaped.

(Also, `Size(dims...)(array)` acheives the same thing)
"""
immutable SizedArray{S,T,N,M} <: StaticArray{T,N}
    data::Array{T,M}

    function SizedArray(a)
        if length(a) != prod(S)
            error("Dimensions $(size(a)) don't match static size $S")
        end
        new(a)
    end
end

@inline (::Type{SizedArray{S,T,N}}){S,T,N,M}(a::Array{T,M}) = SizedArray{S,T,N,M}(a)
@inline (::Type{SizedArray{S,T}}){S,T,M}(a::Array{T,M}) = SizedArray{S,T,_ndims(S),M}(a)
@inline (::Type{SizedArray{S}}){S,T,M}(a::Array{T,M}) = SizedArray{S,T,_ndims(S),M}(a)

# Overide some problematic default behaviour
@inline convert{SA<:SizedArray}(::Type{SA}, sa::SizedArray) = SA(sa.data)

@pure _ndims{N}(::NTuple{N,Int}) = N

@pure size{S}(::Type{SizedArray{S}}) = S
@pure size{S,T}(::Type{SizedArray{S,T}}) = S
@pure size{S,T,N}(::Type{SizedArray{S,T,N}}) = S
@pure size{S,T,N,M}(::Type{SizedArray{S,T,N,M}}) = S

@propagate_inbounds getindex(a::SizedArray, i::Int) = getindex(a.data, i)
@propagate_inbounds setindex!(a::SizedArray, v, i::Int) = setindex!(a.data, v, i)

typealias SizedVector{S,T,M} SizedArray{S,T,1,M}
@inline (::Type{SizedVector{S}}){S,T,M}(a::Array{T,M}) = SizedArray{S,T,1,M}(a)

typealias SizedMatrix{S,T,M} SizedArray{S,T,2,M}
@inline (::Type{SizedMatrix{S}}){S,T,M}(a::Array{T,M}) = SizedArray{S,T,2,M}(a)


"""
    Size(dims)(array)

Creates a `SizedArray` wrapping `array` with the specified statically-known
`dims`, so to take advantage of the (faster) methods defined by the static array
package.
"""
(::Size{S}){S}(a::Array) = SizedArray{S}(a)

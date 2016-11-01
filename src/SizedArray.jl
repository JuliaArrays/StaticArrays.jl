
"""
    SizedArray{(dims...)}(array)

Wraps an `Array` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction.

(Also, `Size(dims...)(array)` acheives the same thing)
"""
immutable SizedArray{S,T,N} <: StaticArray{T,N}
    data::Array{T,N}

    function SizedArray(a)
        if size(a) != S
            error("Dimensions $(size(a)) don't match static size $S")
        end
        new(a)
    end
end

@inline (::Type{SizedArray{S,T}}){S,T,N}(a::Array{T,N}) = SizedArray{S,T,N}(a)
@inline (::Type{SizedArray{S}}){S,T,N}(a::Array{T,N}) = SizedArray{S,T,N}(a)

@pure size{S}(::Type{SizedArray{S}}) = S
@pure size{S,T}(::Type{SizedArray{S,T}}) = S
@pure size{S,T,N}(::Type{SizedArray{S,T,N}}) = S

@propagate_inbounds getindex(a::SizedArray, i::Int) = getindex(a.data, i...)
@propagate_inbounds setindex!(a::SizedArray, v, i::Int) = setindex!(a.data, v, i...)

typealias SizedVector{S,T} SizedArray{S,T,1}
@inline (::Type{SizedVector{S}}){S,T}(a::Vector{T}) = SizedArray{S,T,1}(a)

typealias SizedMatrix{S,T} SizedArray{S,T,2}
@inline (::Type{SizedMatrix{S}}){S,T}(a::Matrix{T}) = SizedArray{S,T,2}(a)


"""
    Size(dims)(array)

Creates a `SizedArray` wrapping `array` with the specified statically-known
`dims`, so to take advantage of the (faster) methods defined by the static array
package.
"""
(::Size{S}){S}(a::Array) = SizedArray{S}(a)

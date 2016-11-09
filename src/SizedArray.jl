
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

    function SizedArray(a::Array)
        if length(a) != prod(S)
            error("Dimensions $(size(a)) don't match static size $S")
        end
        new(a)
    end

    function SizedArray()
        new(Array{T,M}(S))
    end
end

@inline (::Type{SizedArray{S,T,N}}){S,T,N,M}(a::Array{T,M}) = SizedArray{S,T,N,M}(a)
@inline (::Type{SizedArray{S,T}}){S,T,M}(a::Array{T,M}) = SizedArray{S,T,_ndims(S),M}(a)
@inline (::Type{SizedArray{S}}){S,T,M}(a::Array{T,M}) = SizedArray{S,T,_ndims(S),M}(a)

@inline (::Type{SizedArray{S,T,N}}){S,T,N}() = SizedArray{S,T,N,N}()
@inline (::Type{SizedArray{S,T}}){S,T}() = SizedArray{S,T,_ndims(S),_ndims(S)}()

@generated function (::Type{SizedArray{S,T,N,M}}){S,T,N,M,L}(x::NTuple{L})
    if L != prod(S)
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

@inline (::Type{SizedArray{S,T,N}}){S,T,N}(x::Tuple) = SizedArray{S,T,N,N}(x)
@inline (::Type{SizedArray{S,T}}){S,T}(x::Tuple) = SizedArray{S,T,_dims(S),_dims(S)}(x)
@inline (::Type{SizedArray{S}}){S,T,L}(x::NTuple{L,T}) = SizedArray{S,T,_dims(S),_dims(S)}(x)

# Overide some problematic default behaviour
@inline convert{SA<:SizedArray}(::Type{SA}, sa::SizedArray) = SA(sa.data)

# Back to Array (unfortunately need both convert and construct to overide other methods)
@inline (::Type{Array})(sa::SizedArray) = sa.data
@inline (::Type{Array{T}}){T,S}(sa::SizedArray{S,T}) = sa.data
@inline (::Type{Array{T,N}}){T,S,N}(sa::SizedArray{S,T,N}) = sa.data

@inline convert(::Type{Array}, sa::SizedArray) = sa.data
@inline convert{T,S}(::Type{Array{T}}, sa::SizedArray{S,T}) = sa.data
@inline convert{T,S,N}(::Type{Array{T,N}}, sa::SizedArray{S,T,N}) = sa.data

@pure _ndims{N}(::NTuple{N,Int}) = N

@pure size{S}(::Type{SizedArray{S}}) = S
@pure size{S,T}(::Type{SizedArray{S,T}}) = S
@pure size{S,T,N}(::Type{SizedArray{S,T,N}}) = S
@pure size{S,T,N,M}(::Type{SizedArray{S,T,N,M}}) = S

@propagate_inbounds getindex(a::SizedArray, i::Int) = getindex(a.data, i)
@propagate_inbounds setindex!(a::SizedArray, v, i::Int) = setindex!(a.data, v, i)

typealias SizedVector{S,T,M} SizedArray{S,T,1,M}
@pure size{S}(::Type{SizedVector{S}}) = S
@inline (::Type{SizedVector{S}}){S,T,M}(a::Array{T,M}) = SizedArray{S,T,1,M}(a)
@inline (::Type{SizedVector{S}}){S,T,L}(x::NTuple{L,T}) = SizedArray{S,T,1,1}(x)
@inline (::Type{Vector})(sa::SizedVector) = sa.data
@inline convert(::Type{Vector}, sa::SizedVector) = sa.data

typealias SizedMatrix{S,T,M} SizedArray{S,T,2,M}
@pure size{S}(::Type{SizedMatrix{S}}) = S
@inline (::Type{SizedMatrix{S}}){S,T,M}(a::Array{T,M}) = SizedArray{S,T,2,M}(a)
@inline (::Type{SizedMatrix{S}}){S,T,L}(x::NTuple{L,T}) = SizedArray{S,T,2,2}(x)
@inline (::Type{Matrix})(sa::SizedMatrix) = sa.data
@inline convert(::Type{Matrix}, sa::SizedMatrix) = sa.data


"""
    Size(dims)(array)

Creates a `SizedArray` wrapping `array` with the specified statically-known
`dims`, so to take advantage of the (faster) methods defined by the static array
package.
"""
(::Size{S}){S}(a::Array) = SizedArray{S}(a)

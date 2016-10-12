"""
    Scalar{T}(x::T)

Construct a statically-sized 0-dimensional array that contains a single element,
`x`. This type is particularly useful for influencing broadcasting operations.
"""
immutable Scalar{T} <: StaticArray{T,0}
    data::T
end

@inline (::Type{Scalar}){T}(x::Tuple{T}) = Scalar{T}(x[1])

@pure size(::Type{Scalar}) = ()
@pure size{T}(::Type{Scalar{T}}) = ()

@inline function getindex(v::Scalar)
    v.data
end

@inline Tuple(v::Scalar) = (v.data,)

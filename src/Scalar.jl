"""
    Scalar{T}(x::T)

Construct a statically-sized 0-dimensional array that contains a single element,
`x`. This type is particularly useful for influencing broadcasting operations.
"""
immutable Scalar{T} <: StaticArray{T,0}
    data::T
end

@inline (::Type{Scalar{T}}){T}(x::Tuple{T}) = Scalar{T}(x[1])

similar_type{T,S}(::Type{Scalar{T}}, ::Type{S}) = Scalar{S}

@pure size(::Type{Scalar}) = ()
@pure size{T}(::Type{Scalar{T}}) = ()

getindex(v::Scalar) = v.data
@inline function getindex(v::Scalar, i::Int)
    @boundscheck if i != 1
        error("Attempt to index Scalar at index $i")
    end
    v.data
end

@inline Tuple(v::Scalar) = (v.data,)

# A lot more compact than the default array show
Base.show{T}(io::IO, ::MIME"text/plain", x::Scalar{T}) = print(io, "Scalar{$T}(", x.data, ")")

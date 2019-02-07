"""
    MScalar{T}(x::T)

Construct a mutable, statically-sized, 0-dimensional array that contains a
single element, `x`. This type is particularly useful for influencing
broadcasting operations.
"""
const MScalar{T} = MArray{Tuple{},T,0,1}

@inline MScalar(x::Tuple{T}) where {T} = MScalar{T}(x[1])
@inline MScalar(a::AbstractArray) = MScalar{typeof(a)}((a,))
@inline MScalar(a::AbstractScalar) = MScalar{eltype(a)}((a[],)) # Do we want this to convert or wrap?
@inline function convert(::Type{SA}, a::AbstractArray) where {SA <: MScalar}
    return MScalar((a[],))
end
@inline convert(::Type{SA}, sa::SA) where {SA <: MScalar} = sa

getindex(v::MScalar) = v[1]
setindex!(v::MScalar, x) = v[1] = x

# A lot more compact than the default array show
Base.show(io::IO, ::MIME"text/plain", x::MScalar{T}) where {T} = print(io, "MScalar{$T}(", x.data, ")")

# Simplified show for the type
show(io::IO, ::Type{MScalar{T}}) where {T} = print(io, "MScalar{T}")

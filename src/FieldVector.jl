"""
    abstract FieldVector{T} <: StaticVector{T}

Inheriting from this type will make it easy to create your own vector types.
A `FieldVector` will automatically determine its size from the number of fields
(or it can be overriden by `size()`), and define `getindex` and `setindex!`
appropriately. An immutable `FieldVector` will be as performant as an `SVector`
of similar length and element type, while a mutable `FieldVector` will behave
similarly to an `MVector`.

For example:

    immutable/type Point3D <: FieldVector{Float64}
        x::Float64
        y::Float64
        z::Float64
    end
"""
abstract FieldVector{T} <: StaticVector{T}

# Is this a good idea?? Should people just define constructors that accept tuples?
@inline (::Type{FV}){FV<:FieldVector}(x::Tuple) = FV(x...)

@pure size{FV<:FieldVector}(::Union{FV,Type{FV}}) = (length(FV.types),)
@pure length{FV<:FieldVector}(::Union{FV,Type{FV}}) = size(FV)[1]

@inline getindex(v::FieldVector, i::Integer) = getfield(v, i)
@inline setindex!(v::FieldVector, x, i::Integer) = setfield!(v, i, x)

@inline function Base.unsafe_convert{T}(::Type{Ptr{T}}, v::FieldVector{T})
    Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(v))
end

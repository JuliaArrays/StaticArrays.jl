"""
    abstract FieldVector{T} <: StaticVector{T}

Inheriting from this type will make it easy to create your own vector types. A `FieldVector`
will automatically determine its size from the number of fields, and define `getindex` and
`setindex!` appropriately. An immutable `FieldVector` will be as performant as an `SVector`
of similar length and element type, while a mutable `FieldVector` will behave similarly to
an `MVector`.

For example:

    immutable/type Point3D <: FieldVector{Float64}
        x::Float64
        y::Float64
        z::Float64
    end
"""
abstract type FieldVector{T} <: StaticVector{T} end

# Is this a good idea?? Should people just define constructors that accept tuples?
@inline (::Type{FV}){FV<:FieldVector}(x::Tuple) = FV(x...)

@pure Size{FV<:FieldVector}(::Type{FV}) = Size(nfields(FV))

@inline getindex(v::FieldVector, i::Integer) = getfield(v, i)
@inline setindex!(v::FieldVector, x, i::Integer) = setfield!(v, i, x)

# See #53
Base.cconvert{T}(::Type{Ptr{T}}, v::FieldVector) = Ref(v)
Base.unsafe_convert{T, FV <: FieldVector}(::Type{Ptr{T}}, m::Ref{FV}) =
    _unsafe_convert(Ptr{T}, eltype(FV), m)
_unsafe_convert{T, FV <: FieldVector}(::Type{Ptr{T}}, ::Type{T}, m::Ref{FV}) =
         Ptr{T}(Base.unsafe_convert(Ptr{FV}, m))

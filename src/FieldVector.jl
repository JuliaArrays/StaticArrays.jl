abstract FieldVector{T} <: StaticVector{T}

# Is this a good idea?? Should people just define constructors that accept tuples?
@inline (::Type{FV}){FV<:FieldVector}(x::Tuple) = FV(x...)

@pure size{FV<:FieldVector}(::Union{FV,Type{FV}}) = (length(FV.types),)
@pure length{FV<:FieldVector}(::Union{FV,Type{FV}}) = length(FV.types)

@inline getindex(v::FieldVector, i::Integer) = getfield(v, i)


# For example:
#
# immutable Point3D <: FieldVector{Float64}
#     x::Float64
#     y::Float64
#     z::Float64
# end

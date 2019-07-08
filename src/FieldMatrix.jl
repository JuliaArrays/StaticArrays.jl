"""
    abstract FieldMatrix{N, M, T} <: StaticMatrix{N, M, T}

Inheriting from this type will make it easy to create your own rank-two tensor types. A `FieldMatrix`
will automatically define `getindex` and `setindex!` appropriately. An immutable
`FieldMatrix` will be as performant as an `SMatrix` of similar length and element type,
while a mutable `FieldMatrix` will behave similarly to an `MMatrix`.

For example:

    struct Stress <: FieldMatrix{3, 3, Float64}
        xx::Float64
        xy::Float64
        xz::Float64
        yx::Float64
        yy::Float64
        yz::Float64
        zx::Float64
        zy::Float64
        zz::Float64
    end
"""
abstract type FieldMatrix{N, M, T} <: StaticMatrix{N, M, T} end

@inline (::Type{FM})(x::Tuple{Vararg{Tuple{Vararg{Any, M}}, N}}) where {N, M, FM <: FieldMatrix{N, M}} = FM(Tuple(x[j][i] for i = 1:N, j = 1:M))
@inline (::Type{FM})(x::Tuple{Vararg{Any, N}}) where {FM <: FieldMatrix, N} = N == length(FM) ? FM(x...) : throw(DimensionMismatch("No precise constructor for $FM found. Length of input was $(length(x))."))

@propagate_inbounds Base.getindex(m::FieldMatrix, i::Int) = getfield(m, i)
@propagate_inbounds Base.setindex!(m::FieldMatrix, v, i::Int) = setfield!(m, i, v)

Base.cconvert(::Type{<:Ptr}, m::FieldMatrix) = Base.RefValue(m)
Base.unsafe_convert(::Type{Ptr{T}}, m::Base.RefValue{FM}) where {N,M,T,FM<:FieldMatrix{N,M,T}} = Ptr{T}(Base.unsafe_convert(Ptr{FM}, m))

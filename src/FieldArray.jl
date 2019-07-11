"""
    abstract FieldArray{N, T, D} <: StaticArray{N, T, D}

Inheriting from this type will make it easy to create your own rank-D tensor types. A `FieldArray`
will automatically define `getindex` and `setindex!` appropriately. An immutable
`FieldArray` will be as performant as an `SArray` of similar length and element type,
while a mutable `FieldArray` will behave similarly to an `MArray`.

For example:

    struct Stiffness <: FieldArray{Tuple{2,2,2,2}, Float64, 4}
        xxxx::Float64
        yxxx::Float64
        xyxx::Float64
        yyxx::Float64
        xxyx::Float64
        yxyx::Float64
        xyyx::Float64
        yyyx::Float64
        xxxy::Float64
        yxxy::Float64
        xyxy::Float64
        yyxy::Float64
        xxyy::Float64
        yxyy::Float64
        xyyy::Float64
        yyyy::Float64
    end

 Note that you must define the fields of any `FieldArray` subtype in column major order. If you
 want to use an alternative ordering you will need to pay special attention in providing your
 own definitions of `getindex`, `setindex!` and tuple conversion.
"""
abstract type FieldArray{N, T, D} <: StaticArray{N, T, D} end

"""
    abstract FieldMatrix{N1, N2, T} <: FieldArray{Tuple{N1, N2}, 2}

Inheriting from this type will make it easy to create your own rank-two tensor types. A `FieldMatrix`
will automatically define `getindex` and `setindex!` appropriately. An immutable
`FieldMatrix` will be as performant as an `SMatrix` of similar length and element type,
while a mutable `FieldMatrix` will behave similarly to an `MMatrix`.

For example:

    struct Stress <: FieldMatrix{3, 3, Float64}
        xx::Float64
        yx::Float64
        zx::Float64
        xy::Float64
        yy::Float64
        zy::Float64
        xz::Float64
        yz::Float64
        zz::Float64
    end

 Note that the fields of any subtype of `FieldMatrix` must be defined in column major order.
 This means that formatting of constructors for literal `FieldMatrix` can be confusing. For example

    sigma = Stress(1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0)

    3×3 Stress:
     1.0  4.0  7.0
     2.0  5.0  8.0
     3.0  6.0  9.0


will give you the transpose of what the multi-argument formatting suggests. For clarity,
you may consider using the alternative

    sigma = Stress(@SArray[1.0 2.0 3.0;
                           4.0 5.0 6.0;
                           7.0 8.0 9.0])
"""
abstract type FieldMatrix{N1, N2, T} <: FieldArray{Tuple{N1, N2}, T, 2} end

"""
    abstract FieldVector{N, T} <: FieldArray{Tuple{N}, 1}

Inheriting from this type will make it easy to create your own vector types. A `FieldVector`
will automatically define `getindex` and `setindex!` appropriately. An immutable
`FieldVector` will be as performant as an `SVector` of similar length and element type,
while a mutable `FieldVector` will behave similarly to an `MVector`.

For example:

    struct Point3D <: FieldVector{3, Float64}
        x::Float64
        y::Float64
        z::Float64
    end
"""
abstract type FieldVector{N, T} <: FieldArray{Tuple{N}, T, 1} end

@inline function (::Type{FA})(x::Tuple{Vararg{Any, N}}) where {N, FA <: FieldArray}
   if length(FA) == length(x)
      FA(x...)
   else
      throw(DimensionMismatch("No precise constructor for $FA found. Length of input was $(length(x))."))
   end
end

@propagate_inbounds getindex(a::FieldArray, i::Int) = getfield(a, i)
@propagate_inbounds setindex!(a::FieldArray, x, i::Int) = setfield!(a, i, x)

Base.cconvert(::Type{<:Ptr}, a::FieldArray) = Base.RefValue(a)
Base.unsafe_convert(::Type{Ptr{T}}, m::Base.RefValue{FA}) where {N,T,D,FA<:FieldArray{N,T,D}} =
    Ptr{T}(Base.unsafe_convert(Ptr{FA}, m))

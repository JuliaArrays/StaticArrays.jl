"""
    abstract FieldArray{N, T, D} <: StaticArray{N, T, D}

Inheriting from this type will make it easy to create your own rank-D tensor types. A `FieldArray`
will automatically define `getindex` and `setindex!` appropriately. An immutable
`FieldArray` will be as performant as an `SArray` of similar length and element type,
while a mutable `FieldArray` will behave similarly to an `MArray`.

Note that you must define the fields of any `FieldArray` subtype in column major order. If you
want to use an alternative ordering you will need to pay special attention in providing your
own definitions of `getindex`, `setindex!` and tuple conversion.

If you define a `FieldArray` which is parametric on the element type you should
consider defining `similar_type` as in the `FieldVector` example.


# Example

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
"""
abstract type FieldArray{N, T, D} <: StaticArray{N, T, D} end

"""
    abstract FieldMatrix{N1, N2, T} <: FieldArray{Tuple{N1, N2}, 2}

Inheriting from this type will make it easy to create your own rank-two tensor types. A `FieldMatrix`
will automatically define `getindex` and `setindex!` appropriately. An immutable
`FieldMatrix` will be as performant as an `SMatrix` of similar length and element type,
while a mutable `FieldMatrix` will behave similarly to an `MMatrix`.

Note that the fields of any subtype of `FieldMatrix` must be defined in column
major order unless you are willing to implement your own `getindex`.

If you define a `FieldMatrix` which is parametric on the element type you
should consider defining `similar_type` as in the `FieldVector` example.

# Example

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

    sigma = Stress(SA[1.0 2.0 3.0;
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

If you define a `FieldVector` which is parametric on the element type you
should consider defining `similar_type` to preserve your array type through
array operations as in the example below.

# Example

    struct Vec3D{T} <: FieldVector{3, T}
        x::T
        y::T
        z::T
    end

    StaticArrays.similar_type(::Type{<:Vec3D}, ::Type{T}, s::Size{(3,)}) where {T} = Vec3D{T}
"""
abstract type FieldVector{N, T} <: FieldArray{Tuple{N}, T, 1} end

@inline (::Type{FA})(x::Tuple) where {FA <: FieldArray} = construct_type(FA, x)(x...)

function construct_type(::Type{FA}, x) where {FA <: FieldArray}
    has_size(FA) || error("$FA has no static size!")
    length_match_size(FA, x)
    return adapt_eltype(FA, x)
end

@propagate_inbounds getindex(a::FieldArray, i::Int) = getfield(a, i)
@propagate_inbounds setindex!(a::FieldArray, x, i::Int) = (setfield!(a, i, x); a)

Base.cconvert(::Type{<:Ptr}, a::FieldArray) = Base.RefValue(a)
Base.unsafe_convert(::Type{Ptr{T}}, m::Base.RefValue{FA}) where {N,T,D,FA<:FieldArray{N,T,D}} =
    Ptr{T}(Base.unsafe_convert(Ptr{FA}, m))

# We can preserve FieldArrays in array operations which do not change their `Size` and `eltype`.
# FieldArrays with parametric `eltype` would be adapted to the new `eltype` automatically.
# Otherwise, we fallback to `S/MArray` based on it's mutability.
function similar_type(::Type{A}, ::Type{T}, S::Size) where {T,A<:FieldArray}
    A′ = Base.typeintersect(base_type(A), StaticArray{Tuple{Tuple(S)...},T,length(S)})
    isabstracttype(A′) || A′ === Union{} || return A′
    if ismutabletype(A)
        return mutable_similar_type(T, S, length_val(S))
    else
        return default_similar_type(T, S, length_val(S))
    end
end

# return `Union{}` for Union Type. Otherwise return the constructor with no parameters.
@pure base_type(@nospecialize(T::Type)) = (T′ = Base.unwrap_unionall(T); T′ isa DataType ? T′.name.wrapper : Union{})
if VERSION < v"1.7"
    @pure ismutabletype(@nospecialize(T::Type)) = (T′ = Base.unwrap_unionall(T); T′ isa DataType && T′.mutable)
end

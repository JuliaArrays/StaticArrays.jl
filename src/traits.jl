"""
    Dynamic()

Used to signify that a dimension of an array is not known statically.
"""
struct Dynamic end

const StaticDimension = Union{Int, Dynamic}

"""
    dimmatch(x::StaticDimension, y::StaticDimension)

Return whether dimensions `x` and `y` match at compile time, that is:

* if `x` and `y` are both `Int`s, check that they are equal
* if `x` or `y` are `Dynamic()`, return true
"""
function dimmatch end
@inline dimmatch(x::Int, y::Int) = x === y
@inline dimmatch(x::StaticDimension, y::StaticDimension) = true

"""
    Size(dims::Int...)

`Size` is used extensively throughout the `StaticArrays` API to describe _compile-time_
knowledge of the size of an array. The dimensions are stored as a type parameter and are
statically propagated by the compiler, resulting in efficient, type-inferrable code. For
example, to create a static matrix of zeros, use `A = zeros(SMatrix{3,3})`. The static
size of `A` can be obtained by `Size(A)`. (rather than `size(zeros(3,3))`, which returns
`Base.Tuple{2,Int}`).

Note that if dimensions are not known statically (e.g., for standard `Array`s),
[`Dynamic()`](@ref) should be used instead of an `Int`.

    Size(a::AbstractArray)
    Size(::Type{T<:AbstractArray})

The `Size` constructor can be used to extract static dimension information from a given
array. For example:

```julia-repl
julia> Size(zeros(SMatrix{3, 4}))
Size(3, 4)

julia> Size(zeros(3, 4))
Size(StaticArrays.Dynamic(), StaticArrays.Dynamic())
```

This has multiple uses, including "trait"-based dispatch on the size of a statically-sized
array. For example:

```julia
det(x::StaticMatrix) = _det(Size(x), x)
_det(::Size{(1,1)}, x::StaticMatrix) = x[1,1]
_det(::Size{(2,2)}, x::StaticMatrix) = x[1,1]*x[2,2] - x[1,2]*x[2,1]
# and other definitions as necessary
```

"""
struct Size{S}
    function Size{S}() where {S}
        new{S::Tuple{Vararg{StaticDimension}}}()
    end
end

@pure Size(s::Tuple{Vararg{StaticDimension}}) = Size{s}()
@pure Size(s::StaticDimension...) = Size{s}()
@pure Size(s::Type{<:Tuple}) = Size{tuple(s.parameters...)}()

Base.show(io::IO, ::Size{S}) where {S} = print(io, "Size", S)

function missing_size_error(::Type{SA}) where SA
    error("""
        The size of type `$SA` is not known.

        If you were trying to construct (or `convert` to) a `StaticArray` you
        may need to add the size explicitly as a type parameter so its size is
        inferrable to the Julia compiler (or performance would be terrible). For
        example, you might try

            m = zeros(3,3)
            SMatrix(m)            # this error
            SMatrix{3,3}(m)       # correct - size is inferrable
            SArray{Tuple{3,3}}(m) # correct, note Tuple{3,3}
        """)
end

Size(a::T) where {T<:AbstractArray} = Size(T)
Size(::Type{SA}) where {SA <: StaticArray} = missing_size_error(SA)
Size(::Type{SA}) where {SA <: StaticArray{S}} where {S<:Tuple} = @isdefined(S) ? Size(S) : missing_size_error(SA)

Size(::Type{Adjoint{T, A}}) where {T, A <: AbstractVecOrMat{T}} = Size(Size(A)[2], Size(A)[1])
Size(::Type{Transpose{T, A}}) where {T, A <: AbstractVecOrMat{T}} = Size(Size(A)[2], Size(A)[1])
Size(::Type{Symmetric{T, A}}) where {T, A <: AbstractMatrix{T}} = Size(A)
Size(::Type{Hermitian{T, A}}) where {T, A <: AbstractMatrix{T}} = Size(A)
Size(::Type{Diagonal{T, A}}) where {T, A <: AbstractVector{T}} = Size(Size(A)[1], Size(A)[1])
Size(::Type{<:LinearAlgebra.AbstractTriangular{T, A}}) where {T,A} = Size(A)

@pure Size(::Type{<:AbstractArray{<:Any, N}}) where {N} = Size(ntuple(_ -> Dynamic(), N))

struct Length{L}
    function Length{L}() where L
        check_length(L)
        new{L}()
    end
end

check_length(L::Int) = nothing
check_length(L::Dynamic) = nothing
check_length(L) = error("Length was expected to be an `Int` or `Dynamic`")

Base.show(io::IO, ::Length{L}) where {L} = print(io, "Length(", L, ")")

Length(a::AbstractArray) = Length(Size(a))
Length(::Type{A}) where {A <: AbstractArray} = Length(Size(A))
@pure Length(L::Int) = Length{L}()
Length(::Size{S}) where {S} = _Length(S...)
@pure _Length(S::Int...) = Length{prod(S)}()
@inline _Length(S...) = Length{Dynamic()}()

# Some @pure convenience functions for `Size`
@pure (::Type{Tuple})(::Size{S}) where {S} = S

@pure getindex(::Size{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1

@pure length(::Size{S}) where {S} = length(S)
@pure length_val(::Size{S}) where {S} = Val{length(S)}

# Note - using === here, as Base doesn't inline == for tuples as of julia-0.6
@pure Base.:(==)(::Size{S}, s::Tuple{Vararg{Int}}) where {S} = S === s
@pure Base.:(==)(s::Tuple{Vararg{Int}}, ::Size{S}) where {S} = s === S

@pure Base.prod(::Size{S}) where {S} = prod(S)

Base.LinearIndices(::Size{S}) where {S} = LinearIndices(S)

@pure size_tuple(::Size{S}) where {S} = Tuple{S...}

# Some @pure convenience functions for `Length`
@pure (::Type{Int})(::Length{L}) where {L} = L

@pure Base.:(==)(::Length{L}, l::Int) where {L} = L == l
@pure Base.:(==)(l::Int, ::Length{L}) where {L} = l == L

# unroll_tuple also works with `Length`
@propagate_inbounds unroll_tuple(f, ::Length{L}) where {L} = unroll_tuple(f, Val{L})

"""
    sizematch(::Size, ::Size)
    sizematch(::Tuple, ::Tuple)

Determine whether two sizes match, in the sense that they have the same
number of dimensions, and their dimensions match as determined by [`dimmatch`](@ref).
"""
@pure sizematch(::Size{S1}, ::Size{S2}) where {S1, S2} = sizematch(S1, S2)
@inline sizematch(::Tuple{}, ::Tuple{}) = true
@inline sizematch(S1::Tuple{Vararg{StaticDimension, N}}, S2::Tuple{Vararg{StaticDimension, N}}) where {N} =
    dimmatch(S1[1], S2[1]) && sizematch(Base.tail(S1), Base.tail(S2))
@inline sizematch(::Tuple{Vararg{StaticDimension}}, ::Tuple{Vararg{StaticDimension}}) = false # mismatch in number of dimensions

"""
    sizematch(::Size, A::AbstractArray)

Determine whether array `A` matches the given size. If `A` is a
`StaticArray`, the check is performed at compile time, otherwise,
the check is performed at runtime.
"""
@inline sizematch(::Size{S}, A::StaticArray) where {S} = sizematch(Size{S}(), Size(A))
@inline sizematch(::Size{S}, A::AbstractArray) where {S} = sizematch(S, size(A))

"""
Return either the statically known Size() or runtime size()
"""
@inline _size(a) = size(a)
@inline _size(a::StaticArray) = Size(a)

# Return static array from a set of arrays
@inline _first_static(a1::StaticArray, as...) = a1
@inline _first_static(a1, as...) = _first_static(as...)
@inline _first_static() = throw(ArgumentError("No StaticArray found in argument list"))

"""
Returns the common Size of the inputs (or else throws a DimensionMismatch)
"""
@inline function same_size(as...)
    s = Size(_first_static(as...))
    _sizes_match(s, as...) || _throw_size_mismatch(as...)
    s
end
@inline _sizes_match(s::Size, a1, as...) = ((s == _size(a1)) ? _sizes_match(s, as...) : false)
@inline _sizes_match(s::Size) = true
@noinline function _throw_size_mismatch(as...)
    throw(DimensionMismatch("Sizes $(map(_size, as)) of input arrays do not match"))
end

# Return the "diagonal size" of a matrix - the minimum of the two dimensions
diagsize(A::StaticMatrix) = diagsize(Size(A))
@pure diagsize(::Size{S}) where {S} = min(S...)

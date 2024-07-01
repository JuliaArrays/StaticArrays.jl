
"""
    dimmatch(x::StaticDimension, y::StaticDimension)

Return whether dimensions `x` and `y` match at compile time, that is:

* if `x` and `y` are both `Int`s, check that they are equal
* if `x` or `y` are `Dynamic()`, return true
"""
function dimmatch end
@inline dimmatch(x::Int, y::Int) = x === y
@inline dimmatch(x::StaticDimension, y::StaticDimension) = true

Size(::Type{Adjoint{T, A}}) where {T, A <: AbstractVecOrMat{T}} = Size(Size(A)[2], Size(A)[1])
Size(::Type{Transpose{T, A}}) where {T, A <: AbstractVecOrMat{T}} = Size(Size(A)[2], Size(A)[1])
Size(::Type{Symmetric{T, A}}) where {T, A <: AbstractMatrix{T}} = Size(A)
Size(::Type{Hermitian{T, A}}) where {T, A <: AbstractMatrix{T}} = Size(A)
Size(::Type{Diagonal{T, A}}) where {T, A <: AbstractVector{T}} = Size(Size(A)[1], Size(A)[1])
Size(::Type{UpperTriangular{T, A}}) where {T,A} = Size(A)
Size(::Type{UnitUpperTriangular{T, A}}) where {T,A} = Size(A)
Size(::Type{LowerTriangular{T, A}}) where {T,A} = Size(A)
Size(::Type{UnitLowerTriangular{T, A}}) where {T,A} = Size(A)

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
Length(L::Int) = Length{L}()
Length(::Size{S}) where {S} = _Length(S...)
_Length(S::Int...) = Length{prod(S)}()
@inline _Length(S...) = Length{Dynamic()}()

# Some convenience functions for `Size`
(::Type{Tuple})(::Size{S}) where {S} = S

getindex(::Size{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1

length(::Size{S}) where {S} = length(S)
length_val(::Size{S}) where {S} = Val{length(S)}

# Note - using === here, as Base doesn't inline == for tuples as of julia-0.6
Base.:(==)(::Size{S}, s::Tuple{Vararg{Int}}) where {S} = S === s
Base.:(==)(s::Tuple{Vararg{Int}}, ::Size{S}) where {S} = s === S

Base.prod(::Size{S}) where {S} = prod(S)

Base.LinearIndices(::Size{S}) where {S} = LinearIndices(S)

size_tuple(::Size{S}) where {S} = Tuple{S...}

# Some convenience functions for `Length`
(::Type{Int})(::Length{L}) where {L} = Int(L)

Base.:(==)(::Length{L}, l::Int) where {L} = L == l
Base.:(==)(l::Int, ::Length{L}) where {L} = l == L

"""
    sizematch(::Size, ::Size)
    sizematch(::Tuple, ::Tuple)

Determine whether two sizes match, in the sense that they have the same
number of dimensions, and their dimensions match as determined by [`dimmatch`](@ref).
"""
sizematch(::Size{S1}, ::Size{S2}) where {S1, S2} = sizematch(S1, S2)
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
    _size(a)

Return either the statically known `Size()` or runtime `size()`
"""
@inline _size(a) = size(a)
@inline _size(a::StaticArray) = Size(a)

# Return static array from a set of arrays
@inline _first_static(a1::StaticArray, as...) = a1
@inline _first_static(a1, as...) = _first_static(as...)
@inline _first_static() = throw(ArgumentError("No StaticArray found in argument list"))

"""
    same_size(as...)

Returns the common `Size` of the inputs (or else throws a `DimensionMismatch`)
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
diagsize(::Size{S}) where {S} = min(S...)

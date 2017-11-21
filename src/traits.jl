"""
    Size(dims::Int...)

`Size` is used extensively in throughout the `StaticArrays` API to describe the size of a
static array desired by the user. The dimensions are stored as a type parameter and are
statically propagated by the compiler, resulting in efficient, type inferrable code. For
example, to create a static matrix of zeros, use `zeros(Size(3,3))` (rather than
`zeros(3,3)`, which constructs a `Base.Array`).

    Size(a::StaticArray)
    Size(::Type{T<:StaticArray})

Extract the `Size` corresponding to the given static array. This has multiple uses,
including using for "trait"-based dispatch on the size of a statically sized array. For
example:
```
det(x::StaticMatrix) = _det(Size(x), x)
_det(::Size{(1,1)}, x::StaticMatrix) = x[1,1]
_det(::Size{(2,2)}, x::StaticMatrix) = x[1,1]*x[2,2] - x[1,2]*x[2,1]
# and other definitions as necessary
```
"""
struct Size{S}
    function Size{S}() where {S}
        new{S::Tuple{Vararg{Int}}}()
    end
end

@pure Size(s::Tuple{Vararg{Int}}) = Size{s}()
@pure Size(s::Int...) = Size{s}()
@pure Size(s::Type{<:Tuple}) = Size{tuple(s.parameters...)}()

Base.show(io::IO, ::Size{S}) where {S} = print(io, "Size", S)

#= There seems to be a subtyping/specialization bug...
function Size(::Type{SA}) where {SA <: StaticArray} # A nice, default error message for when S not defined
    error("""
        The size of type `$SA` is not known.

        If you were trying to construct (or `convert` to) a `StaticArray` you
        may need to add the size explicitly as a type parameter so its size is
        inferrable to the Julia compiler (or performance would be terrible). For
        example, you might try

            m = zeros(3,3)
            SMatrix(m)      # this error
            SMatrix{3,3}(m) # correct - size is inferrable
        """)
end =#
Size(a::StaticArray{S}) where {S} = Size(S)
Size(a::Type{<:StaticArray{S}}) where {S} = Size(S)

struct Length{L}
    function Length{L}() where L
        check_length(L)
        new{L}()
    end
end

check_length(L::Int) = nothing
check_length(L) = error("Length was expected to be an `Int`")

Base.show(io::IO, ::Length{L}) where {L} = print(io, "Length(", L, ")")

Length(a::StaticArray) = Length(Size(a))
Length(::Type{SA}) where {SA <: StaticArray} = Length(Size(SA))
@pure Length(::Size{S}) where {S} = Length{prod(S)}()
@pure Length(L::Int) = Length{L}()

# Some @pure convenience functions for `Size`
@pure get(::Size{S}) where {S} = S

@pure getindex(::Size{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1

@pure length(::Size{S}) where {S} = length(S)
@pure length_val(::Size{S}) where {S} = Val(length(S))

# Note - using === here, as Base doesn't inline == for tuples as of julia-0.6
@pure Base.:(==)(::Size{S}, s::Tuple{Vararg{Int}}) where {S} = S === s
@pure Base.:(==)(s::Tuple{Vararg{Int}}, ::Size{S}) where {S} = s === S

@pure Base.:(!=)(::Size{S}, s::Tuple{Vararg{Int}}) where {S} = S !== s
@pure Base.:(!=)(s::Tuple{Vararg{Int}}, ::Size{S}) where {S} = s !== S

@pure Base.prod(::Size{S}) where {S} = prod(S)

@pure @inline Base.sub2ind(::Size{S}, x::Int...) where {S} = sub2ind(S, x...)

@pure size_tuple(::Size{S}) where {S} = Tuple{S...}

# Some @pure convenience functions for `Length`
@pure get(::Length{L}) where {L} = L

@pure Base.:(==)(::Length{L}, l::Int) where {L} = L == l
@pure Base.:(==)(l::Int, ::Length{L}) where {L} = l == L

@pure Base.:(!=)(::Length{L}, l::Int) where {L} = L != l
@pure Base.:(!=)(l::Int, ::Length{L}) where {L} = l != L

# unroll_tuple also works with `Length`
@propagate_inbounds unroll_tuple(f, ::Length{L}) where {L} = unroll_tuple(f, Val{L})


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

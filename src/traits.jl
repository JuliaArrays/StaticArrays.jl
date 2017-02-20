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
immutable Size{S}
    function Size()
        check_size(S)
        new()
    end
end

check_size(S::Tuple{Vararg{Int}}) = nothing
check_size(S) = error("Size was expected to be a tuple of `Int`s")

@pure Size(s::Tuple{Vararg{Int}}) = Size{s}()
@pure Size(s::Int...) = Size{s}()

@inline Size(a::StaticArray) = Size(typeof(a))

Base.show{S}(io::IO, ::Size{S}) = print(io, "Size", S)

# A nice, default error message
function Size{SA <: StaticArray}(::Type{SA})
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
end

# Some @pure convenience functions.

@pure get{S}(::Size{S}) = S
@pure getindex{S}(::Size{S}, i::Int) = i <= length(S) ? S[i] : 1

@pure length{S}(::Size{S}) = length(S)
@generated length_val{S}(::Size{S}) = Val{length(S)}

@pure Base.:(==){S}(::Size{S}, s::Tuple{Vararg{Int}}) = S == s
@pure Base.:(==){S}(s::Tuple{Vararg{Int}}, ::Size{S}) = s == S

@pure Base.:(!=){S}(::Size{S}, s::Tuple{Vararg{Int}}) = S != s
@pure Base.:(!=){S}(s::Tuple{Vararg{Int}}, ::Size{S}) = s != S

@pure Base.prod{S}(::Size{S}) = prod(S)

@pure @inline Base.sub2ind{S}(::Size{S}, x::Int...) = sub2ind(S, x...)

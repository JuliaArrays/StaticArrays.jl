"""
    Size(static_array)
    Size(StaticArrayType)
    Size(dims...)

A trait type allowing convenient trait-based dispatch on the size of a statically
sized array. The dimensions are stored as a type parameter and are statically
propagated by the compiler.

For example,
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

@pure check_size(S::Tuple{Vararg{Int}}) = nothing
check_size(S) = error("Size was expected to be a tuple of `Int`s")

@pure Size{SA<:StaticArray}(::Type{SA}) = Size{size(SA)}()
@inline Size(a::StaticArray) = Size(typeof(a))

@pure Size(s::Tuple{Vararg{Int}}) = Size{s}()
@pure Size(s::Int...) = Size{s}()

Base.show{S}(io::IO, ::Size{S}) = print(io, "Size", S)


# Some @pure convenience functions.

# (This type could *probably* be returned from the `size()` function.
# This might enable some generic programming, e.g. with `similar(A, size(A))`.)

@pure getindex{S}(::Size{S}, i::Int) = S[i]

@pure Base.:(==){S}(::Size{S}, s::Tuple{Vararg{Int}}) = S == s
@pure Base.:(==){S}(s::Tuple{Vararg{Int}}, ::Size{S}) = s == S

@pure Base.:(!=){S}(::Size{S}, s::Tuple{Vararg{Int}}) = S != s
@pure Base.:(!=){S}(s::Tuple{Vararg{Int}}, ::Size{S}) = s != S

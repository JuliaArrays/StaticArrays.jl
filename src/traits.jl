"""
    SizeOf(static_array)

Returns the size of a static array, wrapped in a `Size` trait such as
`Size{(2,3)}` for a 2×3 matrix.

`SizeOf` implements the "traitor" paradigm for traits, whereby an abstract trait
class `SizeOf` returns a direct subtype, in this case `Size`.

For example,
```
det(x::StaticMatrix) = det(SizeOf(x), x)
det(::Type{Size{1,1}}, x::AbstractMatrix) = x[1,1]
det(::Type{Size{2,2}}, x::AbstractMatrix) = x[1,1]*x[2,2] - x[1,2]*x[2,1]
# and other definitions as necessary
```
"""
abstract SizeOf


"""
    Size{(dims...)} <: SizeOf

A trait type allowing convenient trait-based dispatch on the size of a statically
sized array.

`Size` implements the "traitor" paradigm for traits, whereby an abstract trait
class `SizeOf` returns a direct subtype, in this case `Size`.

For example,
```
det(x::StaticMatrix) = det(SizeOf(x), x)
det(::Type{Size{1,1}}, x::AbstractMatrix) = x[1,1]
det(::Type{Size{2,2}}, x::AbstractMatrix) = x[1,1]*x[2,2] - x[1,2]*x[2,1]
# and other definitions as necessary
```
"""
immutable Size{S} <: SizeOf
end

@pure SizeOf{SA<:StaticArray}(::Type{SA}) = Size{size(SA)}
@inline SizeOf(a::StaticArray) = Size(typeof(a))

# Also define these, since may be more convenient than SizeOf
"""
    Size(static_array)
    Size(StaticArrayType)

Convenience constructor for the `Size` of a static array. See also `SizeOf`.
"""
@pure Size{SA<:StaticArray}(::Type{SA}) = Size{size(SA)}
@inline Size(a::StaticArray) = Size(typeof(a))

"""
    Size((dims...))
    Size(dims...)

Pure function that constructs a compile-time constant `Size` of an array. This
allows for dispatch of standard (dynamically-sized) arrays to faster, specialized
*StaticArrays* library methods when the programmer knows or can reasonably infer
the size. For example,
```
mat = [1.0 2.0; 3.0 4.0]
det(Size(2,2), mat)  # Faster than det(mat)
```
"""
@pure Size(s::Tuple{Vararg{Int}}) = Size{s}
@pure Size(s::Int...) = Size{s}

@pure getindex{S}(::Type{Size{S}}, i::Int) = S[i]

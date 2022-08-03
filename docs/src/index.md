# Static Arrays
*Statically sized arrays for Julia*

**StaticArrays** provides a framework for implementing statically sized arrays
in Julia, using the abstract type `StaticArray{Size,T,N} <: AbstractArray{T,N}`.
Subtypes of [`StaticArray`](@ref) will provide fast implementations of common array and
linear algebra operations. Note that here "statically sized" means that the
size can be determined from the *type*, and "static" does **not** necessarily
imply `immutable`.

The package also provides some concrete static array types: [`SVector`](@ref), [`SMatrix`](@ref)
and [`SArray`](@ref), which may be used as-is (or else embedded in your own type).
Mutable versions [`MVector`](@ref), [`MMatrix`](@ref) and [`MArray`](@ref) are also exported, as well
as [`SizedArray`](@ref) for annotating standard `Array`s with static size information.
Further, the abstract [`FieldVector`](@ref) can be used to make fast static vectors
out of any uniform Julia "struct".


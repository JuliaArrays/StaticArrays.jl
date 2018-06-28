# Static Arrays
*Statically sized arrays for Julia*

**StaticArrays** provides a framework for implementing statically sized arrays
in Julia (≥ 0.5), using the abstract type `StaticArray{Size,T,N} <: AbstractArray{T,N}`.
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

## Migrating code from Julia v0.6 to Julia v0.7

When upgrading code that is depending on **StaticArrays** the following notes may be helpful

* `chol` has been renamed to `cholesky` and return a factorization object. To obtain the factor
  use `C = cholesky(A).U`, just like for regular Julia arrays.

* `lu` now return a factorization object instead of a tuple with `L`, `U`, and `p`.
  They can be obtained by destructing via iteration (`L, U, p = lu(A)`) or by
  using `getfield` (`F = lu(A); L, U, p = F.L, F.U, F.p`).

* `qr` now return a factorization object instead of a tuple with `Q` and `R`.
  They can be obtained by destructing via iteration (`Q, R = qr(A)`) or by
  using `getfield` (`F = qr(A); Q, R = F.Q, F.R`)

* `eig` has been renamed to `eigen`, which return a factorization object, rather than
  a tuple with `(values, vectors)`. They can be obtained by destructing via iteration
  (`values, vectors = eigen(A)`) or by using `getfield`
  (`E = eigen(A); values, vectors = E.values, E.vectors`).

* `unshift` and `shift` have been renamed to `pushfirst` and `popfirst`.
